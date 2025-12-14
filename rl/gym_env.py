# rl/gym_env_v2_enhanced.py
"""
Enhanced Digital Twin Environment v2.0

Integrates:
- Peltier cooling system (env_model_enhanced.py)
- Comprehensive nutrients (nutrient_model.py)  
- Spatial hardware (hardware_spatial.py)
- CO2 modeling
- Multi-zone climate control

This is a DEMONSTRATION of integration - adjust based on your needs.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any

# Import plant model
from sim.plant_SFPM import HydroponicPlantFSPM, PlantParameters

# Import enhanced models
from sim.env_model import EnhancedEnvironmentModel
from sim.nutrient_model import NutrientSolutionModel, NutrientRecipe
from sim.hardware import SpatialHardwareModel

# ============================================================================
# OBSERVATION SCHEMA - AUTHORITATIVE DEFINITION
# ============================================================================
# This schema defines the exact order and meaning of observation features.
# ALL code that decodes observations MUST use this schema.
# 
# Structure: 11 plant + 8 env + 5 nutrients + (5 + n_peltiers) hardware + 2 time = 31 + n_peltiers
# ============================================================================

def _build_obs_keys(n_peltiers: int = 4) -> list:
    """Build observation key list for given number of Peltier modules."""
    return [
        # Plant state (11 features)
        'plant_biomass_fraction',
        'plant_moisture',
        'plant_nutrient',
        'plant_LAI',
        'plant_stress_water',
        'plant_stress_temp',
        'plant_stress_nutrient',
        'plant_height',
        'plant_NSC',
        'plant_N_content',
        'plant_total_biomass',
        # Environment state (8 features)
        'env_T_top',
        'env_T_middle',
        'env_T_bottom',
        'env_temp_stress',
        'env_RH_top',
        'env_RH_middle',
        'env_RH_bottom',
        'env_CO2',
        # Nutrient state (5 features)
        'nutrient_EC',
        'nutrient_pH',
        'nutrient_N_ppm',
        'nutrient_EC_stress',
        'nutrient_pH_stress',
        # Hardware state (5 base + n_peltiers)
        'hw_shield_pos',
        'hw_fan_on',
        'hw_moisture_std',
        'hw_coverage_efficiency',
        'hw_water_efficiency',
    ] + [f'hw_peltier_{i}' for i in range(n_peltiers)] + [
        # Time encoding (2 features)
        'time_hour_sin',
        'time_hour_cos',
    ]

# Default observation keys (will be updated per-instance based on n_peltiers)
OBS_KEYS = _build_obs_keys(4)

# ============================================================================
# ACTION SCHEMA - AUTHORITATIVE DEFINITION
# ============================================================================
# This schema defines action keys and scaling factors.
# ============================================================================

def _build_action_keys(n_peltiers: int = 4, n_nozzles: int = 12) -> dict:
    """Build action key dictionary with scaling factors."""
    return {
        'water_total': {'scale': 0.05, 'clip': (0.0, 1.0), 'unit': 'L'},
        'fan': {'scale': 1.0, 'clip': (0, 1), 'unit': 'binary'},
        'shield_delta': {'scale': 0.2, 'clip': (-1.0, 1.0), 'unit': 'delta'},
        'heater': {'scale': 200.0, 'clip': (0.0, 1.0), 'unit': 'W'},
        'peltier_controls': {'scale': 1.0, 'clip': (-1.0, 1.0), 'unit': 'power', 'n': n_peltiers},
        'dose_N': {'scale': 0.5, 'clip': (0.0, 1.0), 'unit': 'g'},
        'dose_P': {'scale': 0.1, 'clip': (0.0, 1.0), 'unit': 'g'},
        'dose_K': {'scale': 0.3, 'clip': (0.0, 1.0), 'unit': 'g'},
        'pH_adjust': {'scale': 1.0, 'clip': (-1.0, 1.0), 'unit': 'pH_delta'},
        'nozzle_mask': {'scale': 1.0, 'clip': (0, 1), 'unit': 'binary', 'n': n_nozzles},
        'co2_inject': {'scale': 10.0, 'clip': (0.0, 1.0), 'unit': 'L/hour'},
    }

# Default action keys
ACTION_KEYS = _build_action_keys(4, 12)


class EnhancedDigitalTwinEnv(gym.Env):
    """
    Enhanced environment with:
    1. Peltier cooling array
    2. NPK nutrient management with EC/pH
    3. Spatial plant grid and nozzle control
    4. CO2 concentration control
    5. Multi-zone climate
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, cfg: Optional[dict] = None):
        super().__init__()
        self.cfg = cfg or {}
        
        # === COMPONENT INITIALIZATION ===
        
        # Plant model with species parameters
        plant_params = PlantParameters()
        plant_cfg = self.cfg.get('plant')
        if plant_cfg and isinstance(plant_cfg, dict) and 'species_params' in plant_cfg:
            species_params = plant_cfg['species_params']
            if species_params:
                for key, value in species_params.items():
                    if hasattr(plant_params, key):
                        setattr(plant_params, key, value)
        
        initial_biomass = (plant_cfg.get('initial_biomass', 1.0) 
                          if plant_cfg and isinstance(plant_cfg, dict) 
                          else 1.0)
        self.plant = HydroponicPlantFSPM(
            params=plant_params,
            initial_biomass=initial_biomass,
            dt_hours=1.0
        )
        
        # Enhanced environment model with Peltiers
        env_cfg = self.cfg.get('env')
        if env_cfg is None or not isinstance(env_cfg, dict):
            env_cfg = {}
        env_cfg.setdefault('dt', 1.0)
        self.env = EnhancedEnvironmentModel(env_cfg)
        
        # Nutrient system
        nutrient_cfg = self.cfg.get('nutrients')
        if nutrient_cfg is None or not isinstance(nutrient_cfg, dict):
            nutrient_cfg = {}
        if 'recipe' in nutrient_cfg and isinstance(nutrient_cfg.get('recipe'), dict):
            # Convert dict to NutrientRecipe
            recipe_dict = nutrient_cfg.pop('recipe')
            nutrient_cfg['recipe'] = NutrientRecipe(**recipe_dict)
        self.nutrients = NutrientSolutionModel(nutrient_cfg)
        
        # Spatial hardware with nozzles
        hw_cfg = self.cfg.get('hardware')
        if hw_cfg is None or not isinstance(hw_cfg, dict):
            hw_cfg = {}
        self.hw = SpatialHardwareModel(hw_cfg)
        
        # === ACTION SPACE ===
        # Expanded from 4 to 15 actions
        
        # Store for later use
        env_cfg_for_peltiers = self.cfg.get('env') or {}
        self.n_peltiers = env_cfg_for_peltiers.get('n_peltier_modules', 4) if isinstance(env_cfg_for_peltiers, dict) else 4
        n_peltiers = self.n_peltiers
        
        # Get actual number of nozzles from hardware (now that it's initialized)
        n_nozzles = len(self.hw.nozzles)
        if n_nozzles == 0:
            # Fallback if no nozzles created
            n_nozzles = 12
        
        # Store for action parsing
        self.n_nozzles = n_nozzles
        
        # Build observation schema for this instance
        self.OBS_KEYS = _build_obs_keys(self.n_peltiers)
        
        # Build action schema for this instance
        self.ACTION_KEYS = _build_action_keys(self.n_peltiers, self.n_nozzles)
        
        self.action_space = spaces.Dict({
            # Basic controls (legacy)
            'water_total': spaces.Box(0, 1, (1,), dtype=np.float32),
            'fan': spaces.Discrete(2),
            'shield_delta': spaces.Box(-1, 1, (1,), dtype=np.float32),
            'heater': spaces.Box(0, 1, (1,), dtype=np.float32),
            
            # NEW: Peltier array (cooling/heating per module)
            'peltier_controls': spaces.Box(-1, 1, (n_peltiers,), dtype=np.float32),
            
            # NEW: Nutrient dosing
            'dose_N': spaces.Box(0, 1, (1,), dtype=np.float32),
            'dose_P': spaces.Box(0, 1, (1,), dtype=np.float32),
            'dose_K': spaces.Box(0, 1, (1,), dtype=np.float32),
            'pH_adjust': spaces.Box(-1, 1, (1,), dtype=np.float32),
            
            # NEW: Spatial nozzle control (binary mask)
            'nozzle_mask': spaces.MultiBinary(n_nozzles),
            
            # NEW: CO2 injection
            'co2_inject': spaces.Box(0, 1, (1,), dtype=np.float32),
        })
        
        # === OBSERVATION SPACE ===
        # Expanded from 11 to 29+ features
        
        # Calculate observation dimensions from schema
        obs_dims = len(self.OBS_KEYS)
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dims,), dtype=np.float32
        )
        
        # Assertion: schema length must match observation space
        assert len(self.OBS_KEYS) == obs_dims, f"OBS_KEYS length ({len(self.OBS_KEYS)}) != obs_dims ({obs_dims})"
        
        # === STATE TRACKING ===
        self.hour = 6
        self.day = 0
        self.step_count = 0
        self.max_steps = int(self.cfg.get('max_steps', 24 * 14))
        
        # Rewards
        self.cumulative_growth = 0.0
        self.cumulative_water = 0.0
        self.cumulative_energy = 0.0
        
        # Previous state for deltas
        self.prev_biomass = 1.0
        
        # Termination hysteresis tracking
        self.min_survival_steps = int(self.cfg.get('min_survival_steps', 24))  # Grace period
        self.temp_extreme_count = 0  # Consecutive extreme temp steps
        self.flooded_count = 0  # Consecutive flooded steps
        self.ph_toxic_count = 0  # Consecutive pH toxic steps
        self.temp_extreme_threshold = 6  # Steps before termination
        self.flooded_threshold = 8
        self.ph_toxic_threshold = 6
    
    def _get_true_state(self) -> Dict[str, Any]:
        """
        Get true (un-normalized) state for logging/debugging.
        Returns raw values that can be compared to observations.
        """
        plant_state = self.plant.get_state()
        hw_state = self.hw.get_spatial_stress_map()
        moisture_std = float(np.std(hw_state['moisture'])) if len(hw_state['moisture']) > 0 else 0.0
        
        water_efficiency = 0.0
        if hasattr(self.hw, 'water_efficiency') and len(self.hw.water_efficiency) > 0:
            recent_efficiency = self.hw.water_efficiency[-10:]
            water_efficiency = float(np.mean(recent_efficiency)) if recent_efficiency else 0.0
        
        N_ppm = (self.nutrients.N / max(0.1, self.nutrients.volume_L)) * 1000.0
        
        return {
            'plant': {
                'biomass': plant_state['biomass_fraction'],
                'moisture': plant_state['moisture'],
                'nutrient': plant_state['nutrient'],
                'LAI': plant_state['LAI'],
                'stress_water': plant_state['stress_water'],
                'stress_temp': plant_state['stress_temp'],
                'stress_nutrient': plant_state['stress_nutrient'],
                'height': self.plant.organs.plant_height,
                'NSC': self.plant.organs.NSC,
                'N_content': self.plant.organs.N_content,
                'total_biomass': np.sum(self.plant.organs.B_leaf) + np.sum(self.plant.organs.B_stem) + np.sum(self.plant.organs.B_root),
            },
            'env': {
                'T_top': self.env.T_top,
                'T_middle': self.env.T_middle,
                'T_bottom': self.env.T_bottom,
                'RH_top': self.env.RH_top,
                'RH_middle': self.env.RH_middle,
                'RH_bottom': self.env.RH_bottom,
                'CO2': self.env.CO2,
            },
            'nutrients': {
                'EC': self.nutrients.EC,
                'pH': self.nutrients.pH,
                'N_ppm': N_ppm,
            },
            'hardware': {
                'shield_pos': self.hw.shield_pos,
                'fan_on': self.hw.fan_on,
                'moisture_std': moisture_std,
                'coverage_efficiency': self.hw._calculate_coverage_efficiency(),
                'water_efficiency': water_efficiency,
            },
            'peltier_states': [p.power for p in self.env.peltiers],  # Peltier module power levels [-1, 1]
            'time': {
                'hour': self.hour,
                'day': self.day,
            }
        }
    
    def _obs_to_state_dict(self, obs: np.ndarray) -> Dict[str, Any]:
        """
        Convert observation array back to interpretable state dict for logging.
        This shows what the agent actually sees (normalized/clipped values).
        Uses OBS_KEYS schema for correct decoding.
        """
        # Assert observation matches schema
        assert len(obs) == len(self.OBS_KEYS), \
            f"Observation length ({len(obs)}) != OBS_KEYS length ({len(self.OBS_KEYS)})"
        
        # Build dict using schema
        obs_dict = {}
        for i, key in enumerate(self.OBS_KEYS):
            obs_dict[key] = float(obs[i])
        
        # Reverse normalization for interpretable values
        T_ref = 25.0
        T_scale = 10.0  # Default, may vary with curriculum
        
        # Plant state (denormalize where needed)
        plant_obs = {
            'biomass_fraction': obs_dict['plant_biomass_fraction'],
            'moisture': obs_dict['plant_moisture'],
            'nutrient': obs_dict['plant_nutrient'],
            'LAI': obs_dict['plant_LAI'] * 5.0,  # Reverse normalization
            'stress_water': obs_dict['plant_stress_water'],
            'stress_temp': obs_dict['plant_stress_temp'],
            'stress_nutrient': obs_dict['plant_stress_nutrient'],
            'height': obs_dict['plant_height'] * 0.5,
            'NSC': obs_dict['plant_NSC'] * 5.0,
            'N_content': obs_dict['plant_N_content'],
            'total_biomass': obs_dict['plant_total_biomass'] * 50.0,
        }
        
        # Environment (denormalize temperatures)
        env_obs = {
            'T_top': obs_dict['env_T_top'] * T_scale + T_ref,
            'T_middle': obs_dict['env_T_middle'] * T_scale + T_ref,
            'T_bottom': obs_dict['env_T_bottom'] * T_scale + T_ref,
            'temp_stress': obs_dict['env_temp_stress'],
            'RH_top': obs_dict['env_RH_top'] * 100.0,
            'RH_middle': obs_dict['env_RH_middle'] * 100.0,
            'RH_bottom': obs_dict['env_RH_bottom'] * 100.0,
            'CO2': obs_dict['env_CO2'] * 2000.0,
        }
        
        # Nutrients (denormalize)
        nutrients_obs = {
            'EC': obs_dict['nutrient_EC'] * 3.0,
            'pH': obs_dict['nutrient_pH'] * 4.0 + 4.0,
            'N_ppm': obs_dict['nutrient_N_ppm'] * 100.0,
            'EC_stress': obs_dict['nutrient_EC_stress'],
            'pH_stress': obs_dict['nutrient_pH_stress'],
        }
        
        # Hardware (extract peltier states)
        peltier_states = []
        for i in range(self.n_peltiers):
            key = f'hw_peltier_{i}'
            if key in obs_dict:
                # Convert from [0,1] back to [-1,1] power range
                peltier_states.append(obs_dict[key] * 2.0 - 1.0)
            else:
                peltier_states.append(0.0)
        
        hardware_obs = {
            'shield_pos': obs_dict['hw_shield_pos'],
            'fan_on': obs_dict['hw_fan_on'],
            'moisture_std': obs_dict['hw_moisture_std'],
            'coverage_efficiency': obs_dict['hw_coverage_efficiency'],
            'water_efficiency': obs_dict['hw_water_efficiency'],
            'peltier_states': peltier_states,
        }
        
        return {
            'plant': plant_obs,
            'env': env_obs,
            'nutrients': nutrients_obs,
            'hardware': hardware_obs,
            'time': {
                'hour_sin': obs_dict['time_hour_sin'],
                'hour_cos': obs_dict['time_hour_cos'],
            }
        }
    
    def _get_obs(self) -> np.ndarray:
        """
        Construct observation vector with all enhanced features.
        """
        # === PLANT STATE (11 features) ===
        plant_state = self.plant.get_state()
        obs_plant = [
            plant_state['biomass_fraction'],
            plant_state['moisture'],
            plant_state['nutrient'],
            np.clip(plant_state['LAI'] / 5.0, 0.0, 1.0),
            plant_state['stress_water'],
            plant_state['stress_temp'],
            plant_state['stress_nutrient'],
            # Additional plant metrics
            np.clip(self.plant.organs.plant_height / 0.5, 0.0, 1.0),  # Height normalized
            np.clip(self.plant.organs.NSC / 5.0, 0.0, 1.0),  # NSC pool
            np.clip(self.plant.organs.N_content / 1.0, 0.0, 1.0),  # N content
            np.clip((np.sum(self.plant.organs.B_leaf) + np.sum(self.plant.organs.B_stem) + np.sum(self.plant.organs.B_root)) / 50.0, 0.0, 1.0),  # Total biomass
        ]
        
        # === ENVIRONMENT STATE (9 features) ===
        # Redesigned: Provide both raw temperature (normalized) and stress factor
        # Option 1: Simple linear normalization (T-25)/10 clipped to [-1,1]
        T_ref = 25.0
        T_scale = 10.0
        
        def temp_normalize_simple(T):
            """Simple linear normalization: (T-25)/10 clipped to [-1,1]"""
            return np.clip((T - T_ref) / T_scale, -1.0, 1.0)
        
        def temp_stress(T):
            """Temperature stress factor: 1.0 = optimal, 0.0 = extreme"""
            # Optimal range: 20-30°C
            if 20.0 <= T <= 30.0:
                return 1.0
            elif T < 20.0:
                # Linear decrease below 20°C
                return max(0.0, (T - 10.0) / 10.0)
            else:
                # Linear decrease above 30°C
                return max(0.0, 1.0 - (T - 30.0) / 10.0)
        
        # Get curriculum-aware bounds if available
        curriculum_stage = getattr(self, '_curriculum_stage', None)
        if curriculum_stage in ['warmup', 'easy']:
            # Early curriculum: wider bounds, less sensitive
            T_scale_adaptive = 15.0
        else:
            # Later curriculum: narrower bounds, more sensitive
            T_scale_adaptive = 10.0
        
        obs_env = [
            np.clip((self.env.T_top - T_ref) / T_scale_adaptive, -1.0, 1.0),  # Raw T (normalized)
            np.clip((self.env.T_middle - T_ref) / T_scale_adaptive, -1.0, 1.0),
            np.clip((self.env.T_bottom - T_ref) / T_scale_adaptive, -1.0, 1.0),
            temp_stress(self.env.T_middle),  # Temperature stress factor
            self.env.RH_top / 100.0,
            self.env.RH_middle / 100.0,
            self.env.RH_bottom / 100.0,
            np.clip(self.env.CO2 / 2000.0, 0.0, 1.0),
        ]
        
        # === NUTRIENT STATE (5 features) ===
        # Get current nutrient state
        N_ppm = (self.nutrients.N / max(0.1, self.nutrients.volume_L)) * 1000.0
        obs_nutrients = [
            np.clip(self.nutrients.EC / 3.0, 0.0, 1.0),
            np.clip((self.nutrients.pH - 4.0) / 4.0, 0.0, 1.0),
            np.clip(N_ppm / 100.0, 0.0, 1.0),  # N concentration normalized
            np.clip(1.0 - abs(self.nutrients.EC - 1.8) / 1.8, 0.0, 1.0),  # EC stress
            self.nutrients.get_nutrient_availability()['N'],  # pH stress (using N availability)
        ]
        
        # === HARDWARE STATE (5 + n_peltiers) ===
        hw_state = self.hw.get_spatial_stress_map()
        moisture_std = float(np.std(hw_state['moisture'])) if len(hw_state['moisture']) > 0 else 0.0
        
        # Water efficiency: average of recent efficiency values, or 0 if no water used
        water_efficiency = 0.0
        if hasattr(self.hw, 'water_efficiency') and len(self.hw.water_efficiency) > 0:
            # Use recent average (last 10 steps)
            recent_efficiency = self.hw.water_efficiency[-10:]
            water_efficiency = float(np.mean(recent_efficiency)) if recent_efficiency else 0.0
        
        obs_hardware = [
            self.hw.shield_pos,
            1.0 if self.hw.fan_on else 0.0,
            np.clip(moisture_std, 0.0, 1.0),  # Spatial variance
            self.hw._calculate_coverage_efficiency(),
            water_efficiency,  # Water delivery efficiency (0-1)
        ]
        
        # Peltier module states
        peltier_states = [np.clip((p.power + 1.0) / 2.0, 0.0, 1.0) for p in self.env.peltiers]
        # Ensure we have the right number (pad if needed)
        while len(peltier_states) < self.n_peltiers:
            peltier_states.append(0.0)
        obs_hardware.extend(peltier_states[:self.n_peltiers])
        
        # === TIME ENCODING ===
        hour_rad = (self.hour / 24.0) * 2 * np.pi
        obs_time = [
            (np.sin(hour_rad) + 1.0) / 2.0,
            (np.cos(hour_rad) + 1.0) / 2.0,
        ]
        
        # COMBINE ALL
        obs = np.concatenate([
            obs_plant,
            obs_env,
            obs_nutrients,
            obs_hardware,
            obs_time
        ]).astype(np.float32)
        
        # CRITICAL ASSERTIONS: Ensure observation matches schema
        assert obs.shape == self.observation_space.shape, \
            f"Observation shape {obs.shape} != expected {self.observation_space.shape}"
        assert len(self.OBS_KEYS) == obs.shape[0], \
            f"OBS_KEYS length ({len(self.OBS_KEYS)}) != obs length ({obs.shape[0]})"
        
        return obs
    
    def _compute_reward(self, state: Dict) -> float:
        """
        Enhanced reward function with balanced objectives.
        
        Reward structure:
        - Base: survival + growth + photosynthesis + health (6-12 range)
        - Secondary bonuses: EC, pH, uniformity, CO2 (0-3 range)
        - Penalties: severe stress only (-10 each)
        Total range: 6-15 in good conditions
        """
        # === KEEP ORIGINAL BASE STRUCTURE (2.0 + up to 10 = 6-12 range) ===
        r_survival = 2.0
        r_growth = state.get('biomass_gain', 0.0) * 1000.0
        r_photo = state.get('photosynthesis', 0.0) * 5.0
        
        # === STRESS FACTORS ===
        stress = state.get('stress', {})
        stress_water = stress.get('water', 1.0)
        stress_temp = stress.get('temp', 1.0)
        stress_nutrient = stress.get('nutrient', 1.0)
        r_health = (stress_water + stress_temp + stress_nutrient) * 2.0  # Up to +6
        
        # === NEW: Minor adjustments (max +3 total, keep secondary) ===
        
        # Nutrient balance (reduce from 3.0 to 0.5)
        EC = state.get('EC', 1.8)
        EC_error = abs(EC - 1.8) / 1.8
        r_EC = (1.0 - EC_error) * 0.5  # Max +0.5 (was 3.0)
        
        # pH stability (reduce from 2.0 to 0.5)
        pH = state.get('pH', 6.0)
        pH_error = abs(pH - 6.0) / 2.0
        r_pH = (1.0 - pH_error) * 0.5  # Max +0.5 (was 2.0)
        
        # Spatial uniformity (reduce from 2.0 to 1.0)
        uniformity = state.get('water_uniformity', 0.8)
        r_uniformity = uniformity * 1.0  # Max +1.0 (was 2.0)
        
        # CO2 bonus (reduce from 3.0 to 1.0)
        CO2 = state.get('CO2', 400)
        if 800 <= CO2 <= 1200:
            r_CO2 = 1.0  # Was 3.0
        elif 600 <= CO2 <= 1500:
            r_CO2 = 0.3  # Was 1.0
        else:
            r_CO2 = 0.0
        
        # Energy penalty (reduce from 0.05 to 0.01)
        energy_used = state.get('energy_step', 0.0)
        r_energy = -energy_used * 0.01  # Was 0.05
        
        # === PENALTIES: Keep severe stress penalties at -10 ===
        r_penalty = 0.0
        if stress_water < 0.2:
            r_penalty -= 10.0
        if stress_temp < 0.2:
            r_penalty -= 10.0
        
        # REMOVED: EC/pH extreme penalties (redundant with stress factors)
        # These were removed as they're redundant with the stress factors
        
        # Total: 6-12 (original) + 0-3 (new bonuses) = 6-15 range
        reward = (r_survival + r_growth + r_photo + r_health + 
                 r_EC + r_pH + r_uniformity + r_CO2 + 
                 r_energy + r_penalty )
        
        return float(reward)
    
    def step(self, action):
        """
        Execute one environment step with all enhanced systems.
        """
        # === PARSE ACTIONS ===
        # Handle both Dict (from wrapper conversion) and flat array (direct input)
        if isinstance(action, dict):
            # Action is already a Dict (from FlattenDictAction wrapper)
            water_total = float(np.clip(action['water_total'][0] if isinstance(action['water_total'], np.ndarray) else action['water_total'], 0, 1))
            fan_on = bool(action['fan'])
            shield_delta = float(np.clip(action['shield_delta'][0] if isinstance(action['shield_delta'], np.ndarray) else action['shield_delta'], -1, 1))
            heater = float(np.clip(action['heater'][0] if isinstance(action['heater'], np.ndarray) else action['heater'], 0, 1))
            
            peltier_controls = np.clip(action['peltier_controls'], -1, 1)
            
            dose_N = float(np.clip(action['dose_N'][0] if isinstance(action['dose_N'], np.ndarray) else action['dose_N'], 0, 1)) * 0.5  # Max 0.5g
            dose_P = float(np.clip(action['dose_P'][0] if isinstance(action['dose_P'], np.ndarray) else action['dose_P'], 0, 1)) * 0.1
            dose_K = float(np.clip(action['dose_K'][0] if isinstance(action['dose_K'], np.ndarray) else action['dose_K'], 0, 1)) * 0.3
            pH_adjust = float(np.clip(action['pH_adjust'][0] if isinstance(action['pH_adjust'], np.ndarray) else action['pH_adjust'], -1, 1))
            
            nozzle_mask = action['nozzle_mask']
            co2_inject = float(np.clip(action['co2_inject'][0] if isinstance(action['co2_inject'], np.ndarray) else action['co2_inject'], 0, 1)) * 10.0  # L/hour
        else:
            # Action is a flat array - this shouldn't happen if wrapper is working, but handle it
            # This means the wrapper didn't convert it back - we need to reconstruct the Dict
            action = np.array(action, dtype=np.float32).flatten()
            idx = 0
            
            # Reconstruct Dict from flat array (matching FlattenDictAction order)
            water_total = float(np.clip(action[idx], 0, 1))
            idx += 1
            fan_on = bool(np.clip(action[idx], 0, 1) > 0.5)
            idx += 1
            shield_delta = float(np.clip(action[idx], -1, 1))
            idx += 1
            heater = float(np.clip(action[idx], 0, 1))
            idx += 1
            
            peltier_controls = np.clip(action[idx:idx+self.n_peltiers], -1, 1)
            idx += self.n_peltiers
            
            dose_N = float(np.clip(action[idx], 0, 1)) * 0.5
            idx += 1
            dose_P = float(np.clip(action[idx], 0, 1)) * 0.1
            idx += 1
            dose_K = float(np.clip(action[idx], 0, 1)) * 0.3
            idx += 1
            pH_adjust = float(np.clip(action[idx], -1, 1))
            idx += 1
            
            nozzle_mask = (action[idx:idx+self.n_nozzles] > 0.5).astype(np.int32)
            idx += self.n_nozzles
            
            co2_inject = float(np.clip(action[idx], 0, 1)) * 10.0
        
        # === HARDWARE STEP ===
        # FIXED: Water scaling - ensure water_action=1.0 can offset worst-case ET
        # Typical ET for microgreens: ~0.02-0.05 L/hour per plant
        # With multiple plants, total ET can be 0.1-0.3 L/hour
        # Scale water action to provide meaningful control: 0.05 L per unit action
        # This means water_action=1.0 delivers 0.05 L, which can offset moderate ET
        water_scale = 0.05  # L per unit action (can be tuned based on observed ET)
        hw_out = self.hw.step({
            'water': water_total * water_scale,  # Scale to liters
            'nozzle_control': nozzle_mask,
            'fan': int(fan_on),
            'shield': shield_delta,
            'heater': heater
        })
        
        # === NUTRIENT STEP ===
        nutrient_doses = {
            'N': dose_N,
            'P': dose_P,
            'K': dose_K
        }
        
        # Get root biomass for nutrient uptake calculation
        root_biomass = float(np.sum(self.plant.organs.B_root))
        
        # Calculate water used (from hardware output)
        water_used_L = hw_out.get('delivered_water', 0.0)
        
        nut_state = self.nutrients.step(
            root_biomass=root_biomass,
            dt_hours=1.0,
            water_used_L=water_used_L,
            temp=self.env.T_middle,
            nutrient_dose=nutrient_doses if (dose_N > 0 or dose_P > 0 or dose_K > 0) else None,
            pH_control=True  # Auto pH control
        )
        
        # Manual pH adjustment if agent wants control
        if abs(pH_adjust) > 0.1:
            target_pH = 6.0 + pH_adjust * 1.0  # Range 5.0-7.0
            self.nutrients.adjust_pH(target_pH, adjustment_strength=0.3)
        
        # === ENVIRONMENT STEP (with Peltiers) ===
        # Calculate plant CO2 uptake and transpiration from previous step
        # We'll use current plant state to estimate
        current_LAI = self.plant.organs.LAI
        PAR_estimate = self.env.solar_input(self.hour) * 1500.0  # Estimate PAR
        photo_rate = self.plant.photosynthesis(PAR_estimate, self.env.T_middle)
        plant_co2_uptake = photo_rate * 0.001  # Convert g CO2/m²/h to L/hour (rough estimate)
        
        transp_rate = self.plant.transpiration(self.env.T_middle, self.env.RH_middle, current_LAI)
        evapotranspiration = transp_rate  # L/hour
        
        env_out = self.env.step(
            hour=self.hour,
            shield_pos=hw_out.get('shield_pos', 0.0),
            heater_power=heater,
            fan_on=fan_on,
            peltier_controls=peltier_controls,
            plant_co2_uptake=plant_co2_uptake,
            evapotranspiration=evapotranspiration,
            co2_injection=co2_inject
        )
        
        # === PLANT STEP ===
        # Convert nutrient uptake to plant input format
        # The plant model expects normalized nutrient input
        total_nutrient_uptake = sum(nut_state['uptake'].values())
        nutrient_input_normalized = np.clip(total_nutrient_uptake / 0.1, 0.0, 1.0)  # Normalize
        
        plant_out = self.plant.step(
            light=env_out['L'],
            temp=env_out['T_middle'],
            water_input=hw_out.get('delivered_water', 0.0),
            nutrient_input=nutrient_input_normalized,
            RH=env_out['RH_middle'],
            evaporation=env_out.get('evap', 0.0)
        )
        
        # === COMPUTE REWARD ===
        current_biomass = plant_out.get('biomass_total', 0.0)
        biomass_gain = current_biomass - self.prev_biomass
        
        state = {
            'biomass_gain': biomass_gain,
            'photosynthesis': plant_out.get('photosynthesis_rate', 0.0),
            'stress': {
                'water': plant_out.get('stress_water', 1.0),
                'temp': plant_out.get('stress_temp', 1.0),
                'nutrient': nut_state.get('N_stress', 1.0),
            },
            'EC': nut_state.get('EC', 1.8),
            'pH': nut_state.get('pH', 6.0),
            'CO2': env_out.get('CO2', 400.0),
            'water_uniformity': hw_out.get('water_distribution', {}).get('distribution_uniformity', 0.8),
            'energy_step': (self.hw.energy + self.env.energy_cooling + self.env.energy_heating + self.env.energy_fan) / 1000.0  # Convert to kWh
        }
        reward = self._compute_reward(state)
        
        # Update previous biomass
        self.prev_biomass = current_biomass
        
        # === TERMINATION WITH HYSTERESIS ===
        # Use hysteresis to avoid brittle single-step termination
        # Grace period: no termination during first min_survival_steps (apply penalties instead)
        terminated = False
        truncated = False
        death_reason = None
        terminal_penalty = 0.0
        
        # Check if we're in grace period
        in_grace_period = self.step_count < self.min_survival_steps
        
        # Check termination conditions with hysteresis (consecutive step counting)
        T_mid = env_out.get('T_middle', 20.0)
        pH = nut_state.get('pH', 6.0)
        moisture = plant_out.get('soil_moisture', 0.5)
        
        # 1. Plant death (highest priority, immediate)
        if self.plant.is_dead(temp=T_mid):
            terminated = True
            terminal_penalty = -50.0
            death_reason = 'plant_died'
        
        # 2. Extreme temperature (with hysteresis)
        elif not terminated:
            if T_mid > 38.0 or T_mid < 5.0:
                self.temp_extreme_count += 1
                if self.temp_extreme_count >= self.temp_extreme_threshold and not in_grace_period:
                    terminated = True
                    terminal_penalty = -50.0
                    death_reason = 'temperature_extreme'
                elif in_grace_period:
                    # Apply strong penalty during grace period instead of terminating
                    reward -= 20.0
            else:
                self.temp_extreme_count = 0  # Reset counter
        
        # 3. pH toxicity (with hysteresis)
        if not terminated:
            if pH < 4.5 or pH > 7.5:
                self.ph_toxic_count += 1
                if self.ph_toxic_count >= self.ph_toxic_threshold and not in_grace_period:
                    terminated = True
                    terminal_penalty = -30.0
                    death_reason = 'pH_toxicity'
                elif in_grace_period:
                    reward -= 15.0
            else:
                self.ph_toxic_count = 0
        
        # 4. Severe overwatering/flooded (with hysteresis)
        if not terminated:
            if moisture > 0.95:
                self.flooded_count += 1
                if self.flooded_count >= self.flooded_threshold and not in_grace_period:
                    terminated = True
                    terminal_penalty = -50.0
                    death_reason = 'flooded'
                elif in_grace_period:
                    reward -= 20.0
            else:
                self.flooded_count = 0
        
        # Apply single terminal penalty (not additive)
        if terminated:
            reward += terminal_penalty
        
        # === UPDATE STATE ===
        self.step_count += 1
        self.hour = (self.hour + 1) % 24
        if self.hour == 0:
            self.day += 1
        
        if self.step_count >= self.max_steps:
            truncated = True
        
        # Update cumulative tracking
        self.cumulative_growth += plant_out.get('net_growth', 0.0)
        self.cumulative_water += hw_out.get('delivered_water', 0.0)
        self.cumulative_energy += state.get('energy_step', 0.0)
        
        # Get observation
        obs = self._get_obs()
        
        # Store true state for logging/debugging
        true_state = self._get_true_state()
        self._last_true_state = true_state
        
        # Store applied actions (after scaling/clipping) for logging
        applied_action = {
            'water_total': water_total,
            'fan': fan_on,
            'shield_delta': shield_delta,
            'heater': heater,
            'peltier_controls': peltier_controls.copy() if isinstance(peltier_controls, np.ndarray) else peltier_controls,
            'dose_N': dose_N,
            'dose_P': dose_P,
            'dose_K': dose_K,
            'pH_adjust': pH_adjust,
            'nozzle_mask': nozzle_mask.copy() if isinstance(nozzle_mask, np.ndarray) else nozzle_mask,
            'co2_inject': co2_inject,
        }
        
        # Info
        info = {
            'plant': plant_out,
            'env': env_out,
            'nutrients': nut_state,
            'hw': hw_out,
            'death_reason': death_reason,
            'day': self.day,
            'hour': self.hour,
            'cumulative_growth': self.cumulative_growth,
            'cumulative_water': self.cumulative_water,
            'cumulative_energy': self.cumulative_energy,
            'max_steps': self.max_steps,  # For diagnostics
            # Log true vs observed state for debugging
            'true_state': true_state,
            'observed_state': self._obs_to_state_dict(obs),
            'obs_keys': self.OBS_KEYS,  # Schema for decoding
            'action_keys': self.ACTION_KEYS,  # Action schema
            # Log both raw and applied actions
            'raw_action': action if isinstance(action, dict) else None,  # May be None if already flattened
            'applied_action': applied_action,
            # Debug info: true state values (not decoded obs)
            'debug': {
                'T_middle': float(env_out.get('T_middle', self.env.T_middle)),
                'L': float(env_out.get('L', 0.0)),
                'CO2': float(env_out.get('CO2', self.env.CO2)),
                'RH': float(env_out.get('RH_middle', self.env.RH_middle)),
                'soil_moisture': float(plant_out.get('soil_moisture', true_state['plant'].get('moisture', 0.5))),
                'shield_pos': float(hw_out.get('shield_pos', self.hw.shield_pos)),
                'fan_on': bool(hw_out.get('fan_on', self.hw.fan_on)),
            },
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        # Apply curriculum settings from options if provided
        curriculum_cfg = None
        if options is not None and 'curriculum' in options:
            curriculum_cfg = options['curriculum']
        
        # Reset all subsystems
        plant_cfg = self.cfg.get('plant', {}).copy() if isinstance(self.cfg.get('plant'), dict) else {}
        
        # Apply curriculum settings to plant config
        if curriculum_cfg and 'plant' in curriculum_cfg:
            plant_curriculum = curriculum_cfg['plant']
            if 'initial_moisture' in plant_curriculum:
                plant_cfg['initial_moisture'] = plant_curriculum['initial_moisture']
            if 'moisture_noise_std' in plant_curriculum:
                plant_cfg['moisture_noise_std'] = plant_curriculum['moisture_noise_std']
        
        initial_biomass = plant_cfg.get('initial_biomass', 1.0)
        self.plant.reset(initial_biomass=initial_biomass)
        
        # Reinitialize environment
        env_cfg = self.cfg.get('env', {}).copy() if isinstance(self.cfg.get('env'), dict) else {}
        env_cfg.setdefault('dt', 1.0)
        
        # Apply curriculum settings to environment config
        if curriculum_cfg and 'env' in curriculum_cfg:
            env_curriculum = curriculum_cfg['env']
            if 'initial_T' in env_curriculum:
                env_cfg['initial_T'] = env_curriculum['initial_T']
            if 'temp_noise_std' in env_curriculum:
                env_cfg['temp_noise_std'] = env_curriculum['temp_noise_std']
        
        self.env = EnhancedEnvironmentModel(env_cfg)
        
        # Reinitialize nutrients
        nutrient_cfg = self.cfg.get('nutrients', {}).copy() if isinstance(self.cfg.get('nutrients'), dict) else {}
        if 'recipe' in nutrient_cfg and isinstance(nutrient_cfg.get('recipe'), dict):
            recipe_dict = nutrient_cfg.pop('recipe')
            nutrient_cfg['recipe'] = NutrientRecipe(**recipe_dict)
        self.nutrients = NutrientSolutionModel(nutrient_cfg)
        
        # Reinitialize hardware
        hw_cfg = self.cfg.get('hardware', {}).copy() if isinstance(self.cfg.get('hardware'), dict) else {}
        self.hw = SpatialHardwareModel(hw_cfg)
        
        # Reset all actuator internal states
        self.hw.fan_on = False
        self.hw.shield_pos = 0.0
        if hasattr(self, 'env'):
            self.env.fan_on = False
            for peltier in self.env.peltiers:
                peltier.power = 0.0
        
        # Apply curriculum episode length if provided
        if curriculum_cfg and 'episode_length_days' in curriculum_cfg:
            episode_days = curriculum_cfg['episode_length_days']
            self.max_steps = int(episode_days * 24)
        
        # Apply curriculum actuator limits if provided
        if curriculum_cfg and 'actuator_limits' in curriculum_cfg:
            limits = curriculum_cfg['actuator_limits']
            # Store for use in action scaling (if needed)
            self._curriculum_actuator_limits = limits
        
        # Store curriculum stage for observation bounds
        if curriculum_cfg:
            # Determine stage from config (simple heuristic)
            if 'temp_noise_std' in curriculum_cfg.get('env', {}):
                noise_std = curriculum_cfg['env']['temp_noise_std']
                if noise_std < 0.5:
                    self._curriculum_stage = 'warmup'
                elif noise_std < 1.5:
                    self._curriculum_stage = 'easy'
                elif noise_std < 3.0:
                    self._curriculum_stage = 'medium'
                else:
                    self._curriculum_stage = 'hard'
            else:
                self._curriculum_stage = None
        else:
            self._curriculum_stage = None
        
        self.hour = 6
        self.day = 0
        self.step_count = 0
        
        self.cumulative_growth = 0.0
        self.cumulative_water = 0.0
        self.cumulative_energy = 0.0
        self.prev_biomass = initial_biomass
        
        # Reset termination hysteresis counters
        self.temp_extreme_count = 0
        self.flooded_count = 0
        self.ph_toxic_count = 0
        
        # === FIX RESET CONSISTENCY ===
        # Ensure all subsystems are fully initialized before getting observation
        # Force a step to sync all internal states
        self.hw.step({
            'water': 0.0,
            'nozzle_control': np.zeros(self.n_nozzles, dtype=np.int32),
            'fan': 0,
            'shield': 0.0,
            'heater': 0.0
        })
        
        # Get fresh observation that matches actual state (no stale values)
        obs = self._get_obs()
        
        # Store true state for logging
        self._last_true_state = self._get_true_state()
        
        return obs, {}
    
    def render(self):
        """Enhanced render with all system status"""
        plant_state = self.plant.get_state()
        total_biomass = (np.sum(self.plant.organs.B_leaf) + 
                        np.sum(self.plant.organs.B_stem) + 
                        np.sum(self.plant.organs.B_root))
        
        print(f"Day {self.day} Hour {self.hour:02d}")
        print(f"  Plant: Biomass={total_biomass:.2f}g LAI={plant_state['LAI']:.2f}")
        print(f"  Climate: T={self.env.T_middle:.1f}°C (↑{self.env.T_top:.1f} ↓{self.env.T_bottom:.1f}) "
              f"RH={self.env.RH_middle:.0f}% CO2={self.env.CO2:.0f}ppm")
        N_ppm = (self.nutrients.N / self.nutrients.volume_L) * 1000.0
        print(f"  Nutrients: EC={self.nutrients.EC:.2f} pH={self.nutrients.pH:.1f} N={N_ppm:.0f}ppm")
        print(f"  Spatial: {len(self.hw.plants)} plants, {len(self.hw.nozzles)} nozzles")
        total_energy = (self.hw.energy + self.env.energy_cooling + 
                      self.env.energy_heating + self.env.energy_fan) / 1000.0
        print(f"  Energy: Total={total_energy:.3f}kWh")


# === EXAMPLE CONFIGURATION ===
enhanced_config = {
    'env': {
        'width': 0.6,
        'depth': 0.4,
        'height': 0.8,
        'n_peltier_modules': 4,
        'initial_T': 22.0,
        'initial_RH': 60.0,
        'initial_CO2': 400.0,
    },
    'nutrients': {
        'reservoir_volume_L': 10.0,
        'initial_pH': 6.0,
        'recipe': {  # NutrientRecipe parameters
            'N': 150.0,
            'P': 50.0,
            'K': 200.0,
            'Ca': 160.0,
            'Mg': 50.0,
        }
    },
    'hardware': {
        'width': 0.6,
        'depth': 0.4,
        'plant_spacing': 0.05,  # 5cm between plants
        'nozzle_spacing': 0.15,  # 15cm between nozzles
        'nozzle_type': 'mist'
    },
    'plant': {
        'initial_biomass': 1.0,
        'species_params': {
            # Override PlantParameters if needed
        }
    }
}


if __name__ == '__main__':
    # Test environment
    env = EnhancedDigitalTwinEnv(cfg=enhanced_config)
    
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    
    obs, info = env.reset()
    print(f"\nInitial obs shape: {obs.shape}")
    
    # Random action test
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.2f}")
        
        if terminated or truncated:
            break
    
    print("\nEnvironment test complete!")