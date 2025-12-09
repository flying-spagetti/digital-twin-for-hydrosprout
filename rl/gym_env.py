# rl/gym_env.py
"""
Fixed DigitalTwinEnv (Gymnasium)
- Refactored to prevent "Fan Trap" (cooling without watering).
- Adds specific "Panic Penalties" for drought.
- Adds direct action incentives for watering when dry.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple

from sim.plant_adapter import PlantStructuralAdapter
from sim.env_model import EnvironmentModel
from sim.hardware import HardwareModel
from sim.sensors import SensorModel

# --- Defaults you can tune ---
DEFAULT_MAX_WATER_L = 0.05    # maximum liters delivered if action=1.0 (50 ml)
DEFAULT_MAX_SHIELD_DELTA = 0.15  # max shield position change per step (normalized)
MAX_TEMP_TERMINATE = 38.0
OVERWATER_TERMINATE_THRESH = 0.95
OVERWATER_WARN_THRESH = 0.85
MOISTURE_OPT = 0.50
MOISTURE_BAND = 0.15
DEATH_PENALTY = -1000.0

class DigitalTwinEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__()
        self.cfg = cfg.copy() if cfg else {}
        self.use_extended_obs = self.cfg.get('use_extended_obs', False)
        self.include_soil_obs = self.cfg.get('include_soil_obs', False)
        self.include_nutrient_actions = self.cfg.get('include_nutrient_actions', False)

        # Sub-models
        self.plant = PlantStructuralAdapter(self.cfg.get('plant', {}))
        self.env = EnvironmentModel(self.cfg.get('env', {}))
        self.hw = HardwareModel(self.cfg.get('hardware', {}))
        self.sensors = SensorModel(self.cfg.get('sensors', {}))

        # Observation dimension calculation
        # Base: [canopy, moisture, nutrient, mold, temp_scaled, lux, shield_pos] = 7
        obs_dim = 7
        if self.use_extended_obs:
            obs_dim += 5  # + [LAI, leaf_bio, root_bio, transp, RH]
            if self.include_soil_obs:
                obs_dim += 6  # + [pH, EC, N, P, K, Fe]

        low = np.array([0.0] * obs_dim, dtype=np.float32)
        high = np.array([1.0] * obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Action space
        if self.include_nutrient_actions:
            # water, fan, shield_delta, heater, dose_N, dose_P, dose_K
            self.action_space = spaces.Box(
                low=np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        else:
            # water, fan, shield_delta, heater
            self.action_space = spaces.Box(
                low=np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )

        self.hour = 6
        self.step_count = 0
        self.max_steps = int(self.cfg.get('max_steps', 24 * 14))

        self.disable_actions = set()
        self.max_water_l = float(self.cfg.get('max_water_per_step_l', DEFAULT_MAX_WATER_L))
        self.max_shield_delta = float(self.cfg.get('max_shield_delta', DEFAULT_MAX_SHIELD_DELTA))

        self._last_canopy = None
        self._last_moist = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
            super().reset(seed=seed)

            # 1. Apply Curriculum Config
            if options and isinstance(options, dict) and 'curriculum' in options:
                curr_cfg = options['curriculum'] or {}
                
                # MERGE into self.cfg (ensure keys are dicts, not None)
                if 'env' in curr_cfg and curr_cfg['env'] is not None:
                    if 'env' not in self.cfg or self.cfg['env'] is None:
                        self.cfg['env'] = {}
                    self.cfg['env'].update(curr_cfg['env'])
                if 'plant' in curr_cfg and curr_cfg['plant'] is not None:
                    if 'plant' not in self.cfg or self.cfg['plant'] is None:
                        self.cfg['plant'] = {}
                    self.cfg['plant'].update(curr_cfg['plant'])
                if 'max_steps' in curr_cfg:
                    self.max_steps = int(curr_cfg['max_steps'])
                
                # CRITICAL: Force the sub-models to RE-INIT with new config
                # (Previous code might have just updated cfg but not re-created objects)
                self.env = EnvironmentModel(self.cfg.get('env', {}))
                self.plant = PlantStructuralAdapter(self.cfg.get('plant', {}))
                # Print to confirm it's working in logs
                env_temp = self.cfg.get('env', {}).get('initial_T', 'N/A')
                env_noise = self.cfg.get('env', {}).get('temp_noise_std', 'N/A')
                print(f"[Curriculum] Resetting Env with T={env_temp} and Noise={env_noise}")

            else:
                # Standard reset if no curriculum
                self.plant = PlantStructuralAdapter(self.cfg.get('plant', {}))
                self.env = EnvironmentModel(self.cfg.get('env', {}))

            self.hw = HardwareModel(self.cfg.get('hardware', {}))
            self.sensors = SensorModel(self.cfg.get('sensors', {}))

            self.hour = 6
            self.step_count = 0
            self._last_canopy = None
            self._last_moist = None

            return self._get_obs(), {}
    def _env_step_passive(self):
        return self.env.step(
            self.hour,
            shield_pos=self.hw.shield_pos,
            heater_power=0.0,
            fan_on=self.hw.fan_on
        )

    def _get_plant_scalar_states(self) -> Tuple[float, float, float, float]:
        canopy = 0.0
        moisture = 0.35
        nutrient = 0.5
        pmold = 0.0

        if hasattr(self.plant, 'C'): canopy = float(self.plant.C)
        if hasattr(self.plant, 'M'): moisture = float(self.plant.M)
        if hasattr(self.plant, 'N'): nutrient = float(self.plant.N)
        if hasattr(self.plant, 'P_mold'): pmold = float(self.plant.P_mold)

        # Safe clipping to prevent NANs/infinite rewards
        return (
            float(np.clip(canopy, 0.0, 1.0)),
            float(np.clip(moisture, 0.0, 1.0)),
            float(np.clip(nutrient, 0.0, 1.0)),
            float(np.clip(pmold, 0.0, 1.0))
        )

    def _get_obs(self):
        env_s = self._env_step_passive()
        canopy, moisture, nutrient, pmold = self._get_plant_scalar_states()

        obs_list = [
            canopy,
            moisture,
            nutrient,
            pmold,
            float(env_s.get('T', 20.0) / 40.0),
            float(env_s.get('L', 0.0)),
            float(self.hw.shield_pos if hasattr(self.hw, 'shield_pos') else 0.0)
        ]

        if self.use_extended_obs:
            # Check for structural model attributes
            p = getattr(self.plant, 'plant', None)
            if p is not None:
                # LAI
                lai = float(getattr(p, 'LAI', 0.0))
                obs_list.append(float(np.clip(lai / 5.0, 0.0, 1.0)))
                
                # Biomass
                try:
                    leaf_b = float(np.sum(getattr(p, 'B_leaf', np.zeros(1))))
                    root_b = float(np.sum(getattr(p, 'B_root', np.zeros(1))))
                    max_bio = max(1.0, float(getattr(p, 'n', 1) * 10.0))
                    obs_list.append(float(np.clip(leaf_b / (max_bio * 0.5), 0.0, 1.0)))
                    obs_list.append(float(np.clip(root_b / (max_bio * 0.2), 0.0, 1.0)))
                except Exception:
                    obs_list.extend([0.0, 0.0])

                # Transpiration
                transp = float(getattr(p, '_last_transp', 0.0))
                obs_list.append(float(np.clip(transp / 0.1, 0.0, 1.0))) # Approx max 0.1L

                # RH
                rh = float(env_s.get('RH', 60.0))
                obs_list.append(float(np.clip((rh - 30.0) / 60.0, 0.0, 1.0)))

                # Soil Observations
                if self.include_soil_obs:
                    soil = getattr(p, 'soil', None)
                    if soil is not None:
                        s = soil.status() if callable(getattr(soil, 'status', None)) else {}
                        ph = float(s.get('pH', 6.0))
                        ec = float(s.get('ec', 0.0))
                        n_s = float(s.get('soil_N', 0.0))
                        p_s = float(s.get('soil_P', 0.0))
                        k_s = float(s.get('soil_K', 0.0))
                        fe_s = float(s.get('soil_Fe', s.get('Fe', 0.0)))
                        
                        obs_list.extend([
                            float(np.clip((ph - 4.0) / 4.0, 0.0, 1.0)),
                            float(np.clip(ec / 3.0, 0.0, 1.0)),
                            float(np.clip(n_s, 0.0, 1.0)),
                            float(np.clip(p_s, 0.0, 1.0)),
                            float(np.clip(k_s, 0.0, 1.0)),
                            float(np.clip(fe_s, 0.0, 1.0)),
                        ])
                    else:
                        obs_list.extend([0.0] * 6)
            else:
                # Fallback padding if extended requested but not available
                needed = self.observation_space.shape[0] - len(obs_list)
                obs_list.extend([0.0] * needed)

        return np.array(obs_list, dtype=np.float32)

    def step(self, action):
        action = np.array(action, copy=True, dtype=float)

        # Apply curriculum disables
        if 'water' in self.disable_actions: action[0] = 0.0
        if 'fan' in self.disable_actions and action.shape[0] > 1: action[1] = 0.0
        if 'shield' in self.disable_actions and action.shape[0] > 2: action[2] = 0.0
        if 'heater' in self.disable_actions and action.shape[0] > 3: action[3] = 0.0

        water_a = float(action[0]) if action.shape[0] >= 1 else 0.0
        fan_a = float(action[1]) if action.shape[0] >= 2 else 0.0
        shield_a = float(action[2]) if action.shape[0] >= 3 else 0.0
        heater_a = float(action[3]) if action.shape[0] >= 4 else 0.0

        # Physical scaling
        water_l = float(np.clip(water_a, 0.0, 1.0)) * self.max_water_l
        fan_on = bool(round(np.clip(fan_a, 0.0, 1.0)))
        shield_delta = float(np.clip(shield_a, -1.0, 1.0)) * self.max_shield_delta
        heater_power = float(np.clip(heater_a, 0.0, 1.0))

        nutrient_dose = None
        if self.include_nutrient_actions and action.shape[0] >= 7:
            nutrient_dose = {
                'N': float(np.clip(action[4], 0.0, 1.0)),
                'P': float(np.clip(action[5], 0.0, 1.0)),
                'K': float(np.clip(action[6], 0.0, 1.0)),
                'micro': None, 'chelated': False
            }

        # 1) Hardware Step
        hw_out = self.hw.step({
            'water': water_l, 'fan': int(fan_on), 'shield': shield_delta, 'heater': heater_power
        })

        # 2) Environment Step
        env_out = self.env.step(
            self.hour,
            shield_pos=self.hw.shield_pos,
            heater_power=hw_out.get('heater_power', heater_power),
            fan_on=self.hw.fan_on
        )

        # 3) Plant Step
        nutrient_input = nutrient_dose if nutrient_dose is not None else 0.0
        plant_out = self.plant.step(
            env_out.get('L', 0.0),
            env_out.get('T', 20.0),
            env_out.get('evap', 0.0),
            water_input=hw_out.get('delivered_water', 0.0),
            nutrient_input=nutrient_input,
            env_state=env_out
        )

        # Capture diagnostic info
        pcore = getattr(self.plant, 'plant', None)
        if pcore is not None and isinstance(plant_out, dict):
            diag = plant_out.get('diagnostics') or plant_out.get('diagnostic') or plant_out
            if isinstance(diag, dict) and 'transp_total_liters' in diag:
                setattr(pcore, '_last_transp', float(diag.get('transp_total_liters', 0.0)))

        obs = self._get_obs()
        canopy, moisture, nutrient, pmold = self._get_plant_scalar_states()

        # State Deltas
        if self._last_canopy is None: canopy_delta = 0.0
        else: canopy_delta = float(canopy - self._last_canopy)
        self._last_canopy = canopy

        # --- REWARD FUNCTION ---
        w_canopy = 10.0
        w_mold = 5.0
        w_energy = 0.0  # FREE ENERGY to encourage fan usage
        w_moist = 3.0
        w_temp = 2.0

        reward = 0.0
        reward += w_canopy * max(0.0, canopy_delta)
        reward -= w_mold * (pmold ** 2)
        reward -= w_energy * float(self.hw.energy)

        # Moisture (Bell Curve)
        m_opt = 0.50
        m_band = 0.15
        moist_dist = abs(moisture - m_opt)
        if moist_dist <= m_band:
            reward += w_moist * np.exp(-((moist_dist) / (m_band/2.0))**2)
        else:
            reward -= w_moist * (moist_dist - m_band) * 1.5

        # Temperature (Bell Curve)
        t_opt = 22.0
        t_band = 5.0
        t_curr = float(env_out.get('T', t_opt))
        t_dist = abs(t_curr - t_opt)
        if t_dist <= t_band:
            reward += w_temp * np.exp(-((t_dist) / (t_band/2.0))**2)
        else:
            reward -= w_temp * (t_dist - t_band) * 1.0

        # --- CRITICAL FIXES ---
        # 1. Panic Penalty for Drought
        if moisture < 0.20:
            reward -= 20.0
        
        # 2. Watering Action Incentive
        if moisture < 0.40 and water_a > 0.1:
            reward += 2.0 

        terminated = False
        truncated = False
        death_reason = None
        
        # Termination conditions
        if moisture >= OVERWATER_TERMINATE_THRESH:
            terminated = True
            reward += DEATH_PENALTY
            death_reason = 'flooded'
        elif t_curr >= MAX_TEMP_TERMINATE:
            terminated = True
            reward += DEATH_PENALTY
            death_reason = 'overheat'
        
        try:
            if hasattr(self.plant, 'is_dead') and self.plant.is_dead(temp=t_curr):
                terminated = True
                reward += DEATH_PENALTY
                death_reason = 'plant_dead_internal'
        except: pass

        self.step_count += 1
        self.hour = (self.hour + 1) % 24
        if self.step_count >= self.max_steps:
            truncated = True

        info = {
            'plant': plant_out,
            'env': env_out,
            'hw': hw_out,
            'action_physical': {
                'water_l': water_l,
                'fan_on': int(self.hw.fan_on),
            },
            'death_reason': death_reason,
        }

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        canopy, moisture, _, _ = self._get_plant_scalar_states()
        print(f"Hour {self.hour} | Step {self.step_count} | Canopy {canopy:.3f} | Moist {moisture:.3f} | T {self.env.T:.2f}")