# rl/gym_env_improved.py
"""
Improved DigitalTwin Environment with:
1. Better reward shaping based on plant physiology
2. Clearer action-consequence coupling
3. Multi-objective optimization (growth + efficiency)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any

from sim.plant_SFPM import HydroponicPlantFSPM, PlantParameters
from sim.env_model import EnvironmentModel
from sim.hardware import HardwareModel


class ImprovedDigitalTwinEnv(gym.Env):
    """
    Improved environment with reward structure that teaches agent:
    1. Water when dry → immediate moisture increase
    2. Light + Water → photosynthesis → growth
    3. Temperature control → optimal growth rate
    4. Balance growth vs resource efficiency
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, cfg: Optional[dict] = None):
        super().__init__()
        self.cfg = cfg or {}
        
        # Initialize plant with FSPM
        # Check if species parameters are provided in config
        plant_params = PlantParameters()
        if 'plant' in self.cfg and 'species_params' in self.cfg['plant']:
            # Apply species-specific parameters
            species_params = self.cfg['plant']['species_params']
            for key, value in species_params.items():
                if hasattr(plant_params, key):
                    setattr(plant_params, key, value)
        
        self.plant = HydroponicPlantFSPM(
            params=plant_params,
            initial_biomass=1.0,
            dt_hours=1.0
        )
        
        # Environment and hardware
        self.env = EnvironmentModel(self.cfg.get('env', {}))
        self.hw = HardwareModel(self.cfg.get('hardware', {}))
        
        # Observation space: [biomass_norm, moisture, nutrient, LAI_norm, temp_norm, light_norm, 
        #                     stress_water, stress_temp, stress_nutrient, hour_sin, hour_cos]
        self.observation_space = spaces.Box(
            low=np.array([0.0] * 11, dtype=np.float32),
            high=np.array([1.0] * 11, dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: [water, fan, shield_delta, heater]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self.hour = 6
        self.day = 0
        self.step_count = 0
        self.max_steps = int(self.cfg.get('max_steps', 24 * 14))
        
        # Reward tracking
        self.cumulative_growth = 0.0
        self.cumulative_water_used = 0.0
        self.cumulative_energy_used = 0.0
        
        # Previous state for delta calculations
        self.prev_biomass = 1.0
        self.prev_moisture = 0.5
        
        # Action scaling
        self.max_water_l = 0.05  # 50ml per step
        self.max_shield_delta = 0.15
    
    def _get_obs(self) -> np.ndarray:
        """Get observation with plant physiological state"""
        plant_state = self.plant.get_state()
        
        # Environment state
        temp_norm = np.clip(self.env.T / 40.0, 0.0, 1.0)
        # Compute light from solar input and shield position (same logic as env.step())
        Q_sun = self.env.solar_input(self.hour) * (1.0 - self.hw.shield_pos * self.env.shield_factor)
        light_norm = np.clip(Q_sun * (1 - 0.3 * self.hw.shield_pos), 0.0, 1.0)
        
        # Circadian encoding (helps agent learn day/night cycles)
        hour_rad = (self.hour / 24.0) * 2 * np.pi
        hour_sin = (np.sin(hour_rad) + 1.0) / 2.0
        hour_cos = (np.cos(hour_rad) + 1.0) / 2.0
        
        obs = np.array([
            plant_state['biomass_fraction'],
            plant_state['moisture'],
            plant_state['nutrient'],
            np.clip(plant_state['LAI'] / 5.0, 0.0, 1.0),
            temp_norm,
            light_norm,
            plant_state['stress_water'],
            plant_state['stress_temp'],
            plant_state['stress_nutrient'],
            hour_sin,
            hour_cos,
        ], dtype=np.float32)
        
        return obs
    
    def _compute_reward(self, action: np.ndarray, plant_output: Dict[str, Any], 
                       hw_output: Dict[str, Any]) -> float:
        """
        REBALANCED: Make growth + productivity the main objectives
        
        Reward structure:
        - Survival: Base reward for staying alive
        - Growth: Primary objective (long-term biomass gain)
        - Photosynthesis: Immediate productivity feedback
        - Health: Maintain optimal stress factors
        - Penalties: Only for critical failures
        """
        
        # === SURVIVAL: Base Reward ===
        r_survival = 2.0  # Moderate base reward
        
        # === GROWTH: Primary Objective (Long-term) ===
        current_biomass = plant_output['biomass_total']
        biomass_gain = current_biomass - self.prev_biomass
        r_growth = biomass_gain * 1000.0  # High multiplier makes growth very valuable
        
        # === PHOTOSYNTHESIS: Immediate Feedback (Short-term) ===
        # Reward active photosynthesis to guide agent toward productive states
        photo_rate = plant_output.get('photosynthesis_rate', 0.0)
        r_photo = photo_rate * 5.0  # Immediate reward for productivity
        
        # === HEALTH: Maintain Optimal Conditions ===
        stress_water = plant_output['stress_water']
        stress_temp = plant_output['stress_temp']
        stress_nutrient = plant_output['stress_nutrient']
        
        # Use addition (not multiplication) so one bad stress doesn't zero everything
        r_health = (stress_water + stress_temp + stress_nutrient) * 2.0  # Up to +6
        
        # === PENALTIES: Critical Failures Only ===
        r_penalty = 0.0
        
        # Only penalize SEVERE stress (< 0.2 = critical)
        if stress_water < 0.2:
            r_penalty -= 10.0
        if stress_temp < 0.2:
            r_penalty -= 10.0
        
        # === TOTAL REWARD ===
        # Typical ranges per step:
        # r_survival: +2.0
        # r_growth: +0.5 to +2.0 (0.0005-0.002g × 1000)
        # r_photo: +0.5 to +2.0 (0.1-0.4 g CO2/m²/h × 5)
        # r_health: +3.0 to +6.0 (stress factors 0.5-1.0 each)
        # Total: +6 to +12 per step in good conditions
        reward = r_survival + r_growth + r_photo + r_health + r_penalty
        
        self.prev_biomass = current_biomass
        return float(reward)
    
    def step(self, action: np.ndarray):
        action = np.array(action, dtype=float)
        
        # Parse actions
        water_a = np.clip(action[0], 0.0, 1.0)
        fan_a = np.clip(action[1], 0.0, 1.0)
        shield_a = np.clip(action[2], -1.0, 1.0)
        heater_a = np.clip(action[3], 0.0, 1.0)
        
        # Scale to physical units
        water_l = water_a * self.max_water_l
        fan_on = bool(round(fan_a))
        shield_delta = shield_a * self.max_shield_delta
        heater_power = heater_a
        
        # === HARDWARE STEP ===
        hw_out = self.hw.step({
            'water': water_l,
            'fan': int(fan_on),
            'shield': shield_delta,
            'heater': heater_power
        })
        
        # === ENVIRONMENT STEP ===
        env_out = self.env.step(
            self.hour,
            shield_pos=self.hw.shield_pos,
            heater_power=hw_out.get('heater_power', heater_power),
            fan_on=self.hw.fan_on
        )
        
        # === PLANT STEP ===
        plant_out = self.plant.step(
            light=env_out.get('L', 0.0),
            temp=env_out.get('T', 20.0),
            water_input=hw_out.get('delivered_water', 0.0),
            nutrient_input=0.0,  # Could add nutrient actions
            RH=env_out.get('RH', 60.0),
            evaporation=env_out.get('evap', 0.0)
        )
        
        # === COMPUTE REWARD ===
        reward = self._compute_reward(action, plant_out, hw_out)
        
        # === TERMINATION CONDITIONS ===
        terminated = False
        truncated = False
        death_reason = None
        
        # Plant death
        if self.plant.is_dead(temp=env_out.get('T', 20.0)):
            terminated = True
            reward -= 50.0
            death_reason = 'plant_died'
        
        # Severe overwatering
        if plant_out['soil_moisture'] > 0.95:
            terminated = True
            reward -= 50.0
            death_reason = 'flooded'
        
        # Temperature extremes
        if env_out.get('T', 20.0) > 38.0:
            terminated = True
            reward -= 50.0
            death_reason = 'overheat'
        
        # Update tracking
        self.cumulative_growth += plant_out.get('net_growth', 0.0)
        self.cumulative_water_used += water_l
        self.cumulative_energy_used += self.hw.energy
        
        # Time advancement
        self.step_count += 1
        self.hour = (self.hour + 1) % 24
        if self.hour == 0:
            self.day += 1
        
        if self.step_count >= self.max_steps:
            truncated = True
        
        # Get observation
        obs = self._get_obs()
        
        # Info
        info = {
            'plant': plant_out,
            'env': env_out,
            'hw': hw_out,
            'death_reason': death_reason,
            'day': self.day,
            'cumulative_growth': self.cumulative_growth,
            'cumulative_water_used': self.cumulative_water_used,
            'water_use_efficiency': self.cumulative_growth / max(0.01, self.cumulative_water_used),
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        # === APPLY CURRICULUM CONFIG ===
        env_cfg = self.cfg.copy() if self.cfg else {}
        
        if options and 'curriculum' in options:
            curr_cfg = options['curriculum']
            
            # Merge curriculum settings into environment config
            if 'env' in curr_cfg:
                if 'env' not in env_cfg or env_cfg['env'] is None:
                    env_cfg['env'] = {}
                env_cfg['env'].update(curr_cfg['env'])
            
            if 'plant' in curr_cfg:
                if 'plant' not in env_cfg or env_cfg['plant'] is None:
                    env_cfg['plant'] = {}
                env_cfg['plant'].update(curr_cfg['plant'])
        
        # Create environment with curriculum-adjusted config
        env_config = env_cfg.get('env') or {}
        hw_config = env_cfg.get('hardware') or {}
        self.env = EnvironmentModel(env_config)
        self.hw = HardwareModel(hw_config)
        
        # Reset plant with curriculum initial conditions
        plant_cfg = env_cfg.get('plant') or {}
        init_biomass = plant_cfg.get('initial_biomass', 1.0)
        self.plant.reset(initial_biomass=init_biomass)
        
        # Apply initial moisture if specified
        if 'initial_moisture' in plant_cfg:
            self.plant.soil_moisture = plant_cfg['initial_moisture']
        
        # Reset time
        self.hour = 6
        self.day = 0
        self.step_count = 0
        
        # Reset tracking
        self.cumulative_growth = 0.0
        self.cumulative_water_used = 0.0
        self.cumulative_energy_used = 0.0
        self.prev_biomass = init_biomass
        
        return self._get_obs(), {}
        
    def render(self):
        plant_state = self.plant.get_state()
        print(f"Day {self.day} Hour {self.hour:02d} | "
              f"Biomass: {self.plant.organs.B_leaf.sum() + self.plant.organs.B_stem.sum() + self.plant.organs.B_root.sum():.2f}g | "
              f"LAI: {plant_state['LAI']:.2f} | "
              f"Moisture: {plant_state['moisture']:.2f} | "
              f"Temp: {self.env.T:.1f}°C | "
              f"Stress: W={plant_state['stress_water']:.2f} T={plant_state['stress_temp']:.2f}")