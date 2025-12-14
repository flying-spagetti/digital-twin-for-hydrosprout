#!/usr/bin/env python3
"""
main.py - Orchestrator for the Digital Twin project

Usage examples:
    python main.py gen_synth --out synth_images --n 300
    python main.py demo_ppo
    python main.py train_classifier --data synth_images --epochs 5
    python main.py sim_run --steps 48
    python main.py sample_visual --out sample.png

This script expects to be run from the project root (digital-twin/).
"""
import argparse
import os
import sys
import time
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from viz.analyze_sim_logs import main as analyze_sim_logs
from stable_baselines3 import PPO



# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Local modules (from template)
from visionmodel import synth_generator
from rl.train_ppo import main as ppo_main
from visionmodel import train_classifier
from rl.gym_env import EnhancedDigitalTwinEnv as DigitalTwinEnv
from rl.wrappers import FlattenDictAction
from gymnasium import spaces
from config import get_species_config

# Default uploaded image path from session (you can change this)
DEFAULT_REAL_IMAGE = "/mnt/data/A_high-resolution_digital_photograph_showcases_an_.png"

def gen_synth(args):
    out = args.out or "synth_images"
    n = args.n or 200
    print(f"[main] Generating {n} synthetic images into {out} ...")
    synth_generator.gen_dataset(out_dir=out, n=n)
    print("[main] Done.")

def demo_ppo(args):
    print("[main] Running PPO demo training (quick)...")
    # reuse rl/train_ppo.py entrypoint with --demo
    # Note: the train_ppo.main expects parsed args; we call it via subprocess-like interface
    class Args: demo = True
    ppo_main(Args())
    print("[main] PPO demo finished. Model saved as ppo_demo.")

def train_classifier_cmd(args):
    data_dir = args.data or "synth_images"
    epochs = args.epochs or 5
    print(f"[main] Training classifier on '{data_dir}' for {epochs} epochs ...")
    train_classifier.train(data_dir=data_dir, epochs=epochs)
    print("[main] Classifier training done.")
def sim_run(args):
    # Handle --full_episode flag: run exactly env.max_steps
    if hasattr(args, 'full_episode') and args.full_episode:
        # Will set steps after env is created
        steps = None
    else:
        steps = args.steps or 48
    
    # Load species configuration if specified
    env_cfg = {}
    if hasattr(args, 'species') and args.species:
        try:
            species_cfg = get_species_config(args.species)
            print(f"[main] Loading species: {args.species}")
            print(f"[main] Description: {species_cfg.get('description', 'N/A')}")
            
            # Convert species config to plant parameters format
            # Map species_config.yaml parameters to PlantParameters
            plant_params = {}
            
            # SLA: convert from cm²/g to m²/g
            if 'SLA_cm2_per_g' in species_cfg:
                plant_params['SLA'] = species_cfg['SLA_cm2_per_g'] / 10000.0  # cm² to m²
            
            # Allocation ratios -> partition coefficients
            if 'alloc_leaf' in species_cfg:
                plant_params['partition_leaf_early'] = species_cfg['alloc_leaf']
                plant_params['partition_leaf_late'] = species_cfg['alloc_leaf'] * 0.5  # Reduce in late stage
            if 'alloc_stem' in species_cfg:
                plant_params['partition_stem_early'] = species_cfg['alloc_stem']
                plant_params['partition_stem_late'] = species_cfg['alloc_stem'] * 1.5  # Increase in late stage
            if 'alloc_root' in species_cfg:
                plant_params['partition_root_early'] = species_cfg['alloc_root']
                plant_params['partition_root_late'] = species_cfg['alloc_root'] * 0.7  # Reduce in late stage
            
            # A_max -> RGR_max (approximate mapping)
            if 'A_max' in species_cfg:
                plant_params['RGR_max'] = species_cfg['A_max'] * 6.0  # Scale to reasonable RGR
            
            # Nutrient demand -> N_uptake_rate (approximate)
            if 'nutrient_demand' in species_cfg and 'N' in species_cfg['nutrient_demand']:
                plant_params['N_uptake_rate'] = species_cfg['nutrient_demand']['N'] * 0.04
            
            env_cfg['plant'] = {'species_params': plant_params}
            
        except Exception as e:
            print(f"[main] Warning: Could not load species config: {e}")
            print(f"[main] Using default plant parameters")
    
    # Prepare nutrient dose if specified
    nutrient_dose = None
    if hasattr(args, 'dose_nutrients') and (args.dose_nutrients or args.dose_N is not None or args.dose_P is not None or args.dose_K is not None):
        nutrient_dose = {
            'N': args.dose_N if hasattr(args, 'dose_N') and args.dose_N is not None else (0.1 if hasattr(args, 'dose_nutrients') and args.dose_nutrients else 0.0),
            'P': args.dose_P if hasattr(args, 'dose_P') and args.dose_P is not None else (0.05 if hasattr(args, 'dose_nutrients') and args.dose_nutrients else 0.0),
            'K': args.dose_K if hasattr(args, 'dose_K') and args.dose_K is not None else (0.1 if hasattr(args, 'dose_nutrients') and args.dose_nutrients else 0.0),
            'micro': None,  # Can be extended later
            'chelated': False
        }
    
    # Try to load trained model if specified or auto-detect
    model = None
    if args.no_model:
        model_path = None
    else:
        model_path = args.model if hasattr(args, 'model') and args.model else None
        
        if not model_path:
            # Auto-detect best available model
            best_model = ROOT / "ppo_best" / "best_model.zip"
            full_model = ROOT / "ppo_full.zip"
            demo_model = ROOT / "ppo_demo.zip"
            
            if best_model.exists():
                model_path = str(best_model)
    
    env = DigitalTwinEnv(cfg=env_cfg)
    
    # If --full_episode flag is set, use env.max_steps
    if hasattr(args, 'full_episode') and args.full_episode:
        # Get unwrapped env now (before it's potentially wrapped)
        temp_unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        steps = temp_unwrapped.max_steps
    
    # Apply initial nutrient dose if provided (HydroponicPlantFSPM uses solution_N directly)
    if nutrient_dose:
        # For HydroponicPlantFSPM, we can set solution_N directly (normalized 0-1)
        # Weighted average of N, P, K
        combined_nutrient = 0.5 * nutrient_dose['N'] + 0.3 * nutrient_dose['P'] + 0.2 * nutrient_dose['K']
        env.plant.solution_N = min(1.0, max(0.0, combined_nutrient))
        print(f"[main] Applied initial nutrient dose: N={nutrient_dose['N']:.3f}, P={nutrient_dose['P']:.3f}, K={nutrient_dose['K']:.3f} (combined={combined_nutrient:.3f})")
    
    # If model path found, try to load it
    if model_path and Path(model_path).exists():
        try:
            print(f"[main] Loading trained PPO model from {model_path}")
            # Try to detect observation space from model metadata
            # First, try loading without env to check observation space
            try:
                temp_model = PPO.load(model_path)
                model_obs_dim = temp_model.observation_space.shape[0]
                print(f"[main] Model expects observation space dimension: {model_obs_dim}")
                
                # Determine if we need extended observations
                use_extended = getattr(args, 'use_extended_obs', False)
                if model_obs_dim > 7:
                    # Model was trained with extended observations
                    use_extended = True
                    print(f"[main] Model was trained with extended observations, enabling extended mode")
                
                # Preserve existing env_cfg (including species config) and add extended obs flag
                if use_extended:
                    env_cfg['use_extended_obs'] = True
                
                # Recreate env with correct config (species config is already in env_cfg)
                env = DigitalTwinEnv(cfg=env_cfg)
                
                # Re-apply nutrient dose if env was recreated
                if nutrient_dose:
                    combined_nutrient = 0.5 * nutrient_dose['N'] + 0.3 * nutrient_dose['P'] + 0.2 * nutrient_dose['K']
                    env.plant.solution_N = min(1.0, max(0.0, combined_nutrient))
                
                # FIXED: Wrap environment with FlattenDictAction if model expects Box action space
                # The model was trained with FlattenDictAction wrapper, so we need to apply it here too
                # Store reference to unwrapped env for attribute access
                unwrapped_env = env
                if isinstance(env.action_space, spaces.Dict):
                    env = FlattenDictAction(env)
                    print(f"[main] Wrapped environment with FlattenDictAction for model compatibility")
                
                # Now load model with correct environment
                model = PPO.load(model_path, env=env)
                print(f"[main] Successfully loaded trained model")
            except Exception as e2:
                print(f"[main] Error during model inspection: {e2}")
                raise e2
        except Exception as e:
            print(f"[main] Warning: Could not load model {model_path}: {e}")
            print(f"[main] Falling back to naive policy")
            model = None
    else:
        if model_path:
            print(f"[main] Model path specified but not found: {model_path}")
        print(f"[main] No trained model found, using naive policy")
    
    # Setup logging
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"sim_run_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()  # Also print to console
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info(f"SIMULATION RUN STARTED")
    logger.info(f"Steps: {steps}")
    logger.info(f"Log file: {log_file}")
    if model:
        logger.info(f"Using trained PPO model: {model_path}")
    else:
        logger.info("Using naive/heuristic policy")
    if nutrient_dose:
        logger.info(f"Initial nutrient dose: N={nutrient_dose['N']:.3f}, P={nutrient_dose['P']:.3f}, K={nutrient_dose['K']:.3f}")
    logger.info("="*80)
    
    # Get unwrapped environment for attribute access (if wrapped)
    # Use gymnasium's built-in unwrapped property
    unwrapped_env = env.unwrapped
    
    # Reset environment
    obs, info = env.reset()
    logger.info("Environment reset")
    logger.info(f"Initial hour: {unwrapped_env.hour}")
    logger.info(f"Initial step_count: {unwrapped_env.step_count}")
    logger.info(f"Max steps: {unwrapped_env.max_steps}")
    
    # Log initial observation using schema
    obs_keys = getattr(unwrapped_env, 'OBS_KEYS', None)
    # Initialize observation tracking for delta calculations
    prev_obs_dict = {}  # Previous step observation
    curr_obs_dict = {}  # Current step observation (before action)
    next_obs_dict = {}  # Next step observation (after action)
    
    if obs_keys and len(obs) == len(obs_keys):
        # Use schema-based decoding
        obs_dict = {key: float(obs[i]) for i, key in enumerate(obs_keys)}
        curr_obs_dict = obs_dict.copy()  # Store current observation
        prev_obs_dict = obs_dict.copy()  # Initialize prev with current for first step
        logger.info("Initial Observation (using schema):")
        if 'plant_biomass_fraction' in obs_dict:
            logger.info(f"  Plant biomass fraction: {obs_dict['plant_biomass_fraction']:.4f}")
        if 'plant_moisture' in obs_dict:
            logger.info(f"  Plant moisture: {obs_dict['plant_moisture']:.4f}")
        if 'plant_nutrient' in obs_dict:
            logger.info(f"  Plant nutrient: {obs_dict['plant_nutrient']:.4f}")
        # Use debug info for true values (will be available after first step)
        logger.info(f"  Temperature (middle): (will use true value from debug info)")
        logger.info(f"  CO2: (will use true value from debug info)")
        if 'hw_shield_pos' in obs_dict:
            logger.info(f"  Shield position: {obs_dict['hw_shield_pos']:.4f}")
    else:
        # Fallback: log raw observation if schema not available
        logger.info(f"Initial Observation (raw, len={len(obs)}):")
        for i in range(min(len(obs), 10)):
            logger.info(f"  obs[{i}]: {obs[i]:.4f}")
    
    # Log initial plant state
    logger.info("Initial Plant State:")
    plant_state = unwrapped_env.plant.get_state()
    logger.info(f"  Canopy: {plant_state.get('canopy', 0):.4f}")
    logger.info(f"  Moisture: {plant_state.get('moisture', 0):.4f}")
    logger.info(f"  Nutrient: {plant_state.get('nutrient', 0):.4f}")
    logger.info(f"  LAI (Leaf Area Index): {plant_state.get('LAI', 0):.4f}")
    logger.info(f"  Total leaf biomass: {np.sum(unwrapped_env.plant.organs.B_leaf):.4f} g")
    logger.info(f"  Total root biomass: {np.sum(unwrapped_env.plant.organs.B_root):.4f} g")
    logger.info(f"  Total stem biomass: {np.sum(unwrapped_env.plant.organs.B_stem):.4f} g")
    logger.info(f"  Water stress: {plant_state.get('stress_water', 1.0):.4f}")
    logger.info(f"  Temperature stress: {plant_state.get('stress_temp', 1.0):.4f}")
    logger.info(f"  Nutrient stress: {plant_state.get('stress_nutrient', 1.0):.4f}")
    
    # Log initial environment state
    logger.info("Initial Environment State:")
    logger.info(f"  Temperature: {unwrapped_env.env.T_middle:.2f}°C")
    logger.info(f"  Relative Humidity: {unwrapped_env.env.RH_middle:.2f}%")
    logger.info(f"  Light: {unwrapped_env.env.solar_input(unwrapped_env.hour):.4f}")
    
    # Log initial hardware state
    logger.info("Initial Hardware State:")
    logger.info(f"  Shield position: {unwrapped_env.hw.shield_pos:.4f}")
    logger.info(f"  Fan on: {unwrapped_env.hw.fan_on}")
    logger.info(f"  Energy: {unwrapped_env.hw.energy:.4f}")
    
    logger.info("-"*80)
    
    total_reward = 0.0
    action_history = []
    prev_debug_info = {}  # Store previous debug info for delta calculations
    
    for t in range(steps):
        logger.info(f"\n{'='*80}")
        logger.info(f"STEP {t+1}/{steps} | Hour {unwrapped_env.hour}")
        logger.info(f"{'='*80}")
        
        # Log current observation using schema
        obs_keys = getattr(unwrapped_env, 'OBS_KEYS', None)
        if obs_keys and len(obs) == len(obs_keys):
            # Use schema-based decoding
            obs_dict = {key: float(obs[i]) for i, key in enumerate(obs_keys)}
            curr_obs_dict = obs_dict.copy()  # Store current observation (before action)
            logger.info("Current Observation (using schema):")
            if 'plant_biomass_fraction' in obs_dict:
                prev_val = prev_obs_dict.get('plant_biomass_fraction', obs_dict['plant_biomass_fraction'])
                logger.info(f"  Plant biomass: {obs_dict['plant_biomass_fraction']:.4f} (Δ{obs_dict['plant_biomass_fraction']-prev_val:+.4f})")
            if 'plant_moisture' in obs_dict:
                moist = obs_dict['plant_moisture']
                prev_moist = prev_obs_dict.get('plant_moisture', moist)
                logger.info(f"  Moisture: {moist:.4f} (Δ{moist-prev_moist:+.4f})")
            if 'plant_nutrient' in obs_dict:
                nut = obs_dict['plant_nutrient']
                prev_nut = prev_obs_dict.get('plant_nutrient', nut)
                logger.info(f"  Nutrient: {nut:.4f} (Δ{nut-prev_nut:+.4f})")
            # Temperature and CO2 will be logged from debug info (true values)
            if 'hw_shield_pos' in obs_dict:
                shield = obs_dict['hw_shield_pos']
                prev_shield = prev_obs_dict.get('hw_shield_pos', shield)
                logger.info(f"  Shield: {shield:.4f} (Δ{shield-prev_shield:+.4f})")
        else:
            # Fallback for compatibility
            if len(obs) > 0:
                logger.info(f"Current Observation (raw, len={len(obs)}):")
                for i in range(min(len(obs), 5)):
                    logger.info(f"  obs[{i}]: {obs[i]:.4f}")
        
        # Determine action - use trained model if available, else naive policy
        if model:
            # Use trained PPO model
            action, _ = model.predict(obs, deterministic=True)
            # FIXED: Keep as numpy array for FlattenDictAction wrapper
            # The wrapper needs numpy array for slicing and reshaping
            if not isinstance(action, np.ndarray):
                action = np.array(action, dtype=np.float32)
            action_reasons = [f"PPO model decision (deterministic)"]
            logger.info("Using trained PPO model for action selection")
        else:
            # Fallback to naive policy - create Dict action matching environment action space
            action_reasons = []
            
            # Initialize all action components with defaults
            water_total = 0.0
            fan_on = 0  # Discrete: 0=off, 1=on
            shield_delta = 0.0
            heater = 0.0
            n_peltiers = getattr(unwrapped_env, 'n_peltiers', 4)
            peltier_controls = np.zeros(n_peltiers, dtype=np.float32)
            dose_N = 0.0
            dose_P = 0.0
            dose_K = 0.0
            pH_adjust = 0.0
            n_nozzles = getattr(unwrapped_env, 'n_nozzles', 15)
            nozzle_mask = np.zeros(n_nozzles, dtype=np.int32)
            co2_inject = 0.0
            
            # Initialize moist and temp with defaults to prevent UnboundLocalError
            moist = 0.5  # Default normalized moisture
            temp = 25.0  # Default temperature in °C
            
            # Extract values from current observation if available
            if obs_keys and len(obs) == len(obs_keys):
                if 'plant_moisture' in curr_obs_dict:
                    moist = curr_obs_dict['plant_moisture']
                # For temperature, prefer previous step's debug info (true value)
                # Otherwise use normalized observation as fallback
                if prev_debug_info and 'T_middle' in prev_debug_info:
                    temp = prev_debug_info['T_middle']  # Use true value from previous step
                elif 'env_T_middle' in curr_obs_dict:
                    # Use normalized value as fallback (rough denormalization)
                    temp_normalized = curr_obs_dict['env_T_middle']
                    temp = temp_normalized * 10.0 + 25.0
            
            # Apply naive policy rules with guard conditions
            if moist < 0.35:
                water_total = 0.8
                action_reasons.append(f"Low moisture ({moist:.3f} < 0.35)")
            if temp > 28.0:
                fan_on = 1  # Turn fan on
                shield_delta = 0.2  # Open shield slightly
                # Also use Peltier cooling
                peltier_controls = np.full(len(peltier_controls), -0.5, dtype=np.float32)  # Cooling
                action_reasons.append(f"High temperature ({temp:.2f}°C > 28°C)")
            if temp < 16.0:
                heater = 0.6
                action_reasons.append(f"Low temperature ({temp:.2f}°C < 16°C)")
            
            if not action_reasons:
                action_reasons.append("No action needed (all conditions normal)")
            
            # Create Dict action matching environment action space
            action = {
                'water_total': np.array([water_total], dtype=np.float32),
                'fan': fan_on,
                'shield_delta': np.array([shield_delta], dtype=np.float32),
                'heater': np.array([heater], dtype=np.float32),
                'peltier_controls': peltier_controls,
                'dose_N': np.array([dose_N], dtype=np.float32),
                'dose_P': np.array([dose_P], dtype=np.float32),
                'dose_K': np.array([dose_K], dtype=np.float32),
                'pH_adjust': np.array([pH_adjust], dtype=np.float32),
                'nozzle_mask': nozzle_mask,
                'co2_inject': np.array([co2_inject], dtype=np.float32),
            }
        
        # Log action
        logger.info("Action Selected:")
        if isinstance(action, dict):
            logger.info(f"  Water: {action['water_total'][0]:.2f}")
            logger.info(f"  Fan: {action['fan']} ({'ON' if action['fan'] > 0 else 'OFF'})")
            logger.info(f"  Shield delta: {action['shield_delta'][0]:.2f}")
            logger.info(f"  Heater: {action['heater'][0]:.2f}")
        else:
            # Fallback for list actions (shouldn't happen now)
            logger.info(f"  Water: {action[0]:.2f}")
            logger.info(f"  Fan: {action[1]:.2f} ({'ON' if action[1] > 0.5 else 'OFF'})")
            logger.info(f"  Shield delta: {action[2]:.2f}")
            logger.info(f"  Heater: {action[3]:.2f}")
        logger.info(f"  Reasons: {', '.join(action_reasons)}")
        # Store action for history (convert dict to serializable format)
        if isinstance(action, dict):
            action_history.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in action.items()})
        else:
            action_history.append(action.copy() if hasattr(action, 'copy') else list(action))
        
        # Execute step
        obs, rew, terminated, truncated, info = env.step(action)
        total_reward += rew
        
        # Log step results
        logger.info("Step Results:")
        logger.info(f"  Reward: {rew:.4f}")
        logger.info(f"  Cumulative reward: {total_reward:.4f}")
        logger.info(f"  Terminated: {terminated}")
        logger.info(f"  Truncated: {truncated}")
        
        # Log plant state from info
        plant_info = info.get('plant', {})
        plant_state = unwrapped_env.plant.get_state()
        logger.info("Plant State After Step:")
        logger.info(f"  Canopy: {plant_state.get('canopy', 0):.4f}")
        logger.info(f"  Moisture: {plant_state.get('moisture', 0):.4f}")
        logger.info(f"  Nutrient: {plant_state.get('nutrient', 0):.4f}")
        logger.info(f"  LAI (Leaf Area Index): {plant_state.get('LAI', 0):.4f}")
        logger.info(f"  Total leaf biomass: {np.sum(unwrapped_env.plant.organs.B_leaf):.4f} g")
        logger.info(f"  Total root biomass: {np.sum(unwrapped_env.plant.organs.B_root):.4f} g")
        logger.info(f"  Total stem biomass: {np.sum(unwrapped_env.plant.organs.B_stem):.4f} g")
        logger.info(f"  Water stress: {plant_state.get('stress_water', 1.0):.4f}")
        logger.info(f"  Temperature stress: {plant_state.get('stress_temp', 1.0):.4f}")
        logger.info(f"  Nutrient stress: {plant_state.get('stress_nutrient', 1.0):.4f}")
        if plant_info:
            logger.info(f"  Plant info: {plant_info}")
        
        # Log environment state from info
        env_info = info.get('env', {})
        logger.info("Environment State After Step:")
        logger.info(f"  Temperature: {unwrapped_env.env.T_middle:.2f}°C")
        logger.info(f"  Relative Humidity: {env_info.get('RH', unwrapped_env.env.RH_middle):.2f}%")
        logger.info(f"  Light (L): {env_info.get('L', 0):.4f}")
        logger.info(f"  Evaporation: {env_info.get('evap', 0):.4f}")
        if env_info:
            logger.info(f"  Env info: {env_info}")
        
        # Log hardware state from info
        hw_info = info.get('hw', {})
        logger.info("Hardware State After Step:")
        logger.info(f"  Shield position: {unwrapped_env.hw.shield_pos:.4f}")
        logger.info(f"  Fan on: {unwrapped_env.hw.fan_on}")
        logger.info(f"  Energy consumed: {unwrapped_env.hw.energy:.4f}")
        logger.info(f"  Delivered water: {hw_info.get('delivered_water', 0):.4f}")
        logger.info(f"  Heater power: {hw_info.get('heater_power', 0):.4f}")
        # FIXED: Log both coverage metrics separately
        logger.info(f"  Nozzle coverage fraction (geometry): {hw_info.get('nozzle_coverage_fraction', 0):.4f}")
        logger.info(f"  Water delivery efficiency: {hw_info.get('water_delivery_efficiency', 0):.4f}")
        if hw_info:
            logger.info(f"  HW info: {hw_info}")
        
        # Log applied action from env (post-clip/scaled controls)
        applied_action = info.get('applied_action', {})
        if applied_action:
            logger.info("Applied Action (post-clip/scaled):")
            if 'water_total' in applied_action:
                logger.info(f"  Water: {applied_action['water_total']:.4f}")
            if 'fan' in applied_action:
                logger.info(f"  Fan: {applied_action['fan']} ({'ON' if applied_action['fan'] else 'OFF'})")
            if 'shield_delta' in applied_action:
                logger.info(f"  Shield delta: {applied_action['shield_delta']:.4f}")
            if 'heater' in applied_action:
                logger.info(f"  Heater: {applied_action['heater']:.4f}")
            if 'peltier_controls' in applied_action:
                peltiers = applied_action['peltier_controls']
                if isinstance(peltiers, (list, np.ndarray)):
                    logger.info(f"  Peltier controls: {[f'{p:.2f}' for p in peltiers]}")
                else:
                    logger.info(f"  Peltier controls: {peltiers}")
            if 'dose_N' in applied_action:
                logger.info(f"  Dose N: {applied_action['dose_N']:.4f} g")
            if 'dose_P' in applied_action:
                logger.info(f"  Dose P: {applied_action['dose_P']:.4f} g")
            if 'dose_K' in applied_action:
                logger.info(f"  Dose K: {applied_action['dose_K']:.4f} g")
            if 'co2_inject' in applied_action:
                logger.info(f"  CO2 inject: {applied_action['co2_inject']:.4f} L/hour")
        
        # Log debug info (true state values, not decoded obs)
        debug_info = info.get('debug', {})
        if debug_info:
            logger.info("Debug Info (True State):")
            
            T_middle = debug_info.get('T_middle', 0)
            prev_T = prev_debug_info.get('T_middle', T_middle)
            logger.info(f"  T_middle: {T_middle:.2f}°C (Δ{T_middle-prev_T:+.2f}°C)")
            
            L = debug_info.get('L', 0)
            prev_L = prev_debug_info.get('L', L)
            logger.info(f"  L (light): {L:.4f} (Δ{L-prev_L:+.4f})")
            
            CO2 = debug_info.get('CO2', 0)
            prev_CO2 = prev_debug_info.get('CO2', CO2)
            logger.info(f"  CO2: {CO2:.0f} ppm (Δ{CO2-prev_CO2:+.0f} ppm)")
            
            soil_moisture = debug_info.get('soil_moisture', 0)
            prev_soil_moisture = prev_debug_info.get('soil_moisture', soil_moisture)
            logger.info(f"  Soil moisture: {soil_moisture:.4f} (Δ{soil_moisture-prev_soil_moisture:+.4f})")
            
            shield_pos = debug_info.get('shield_pos', 0)
            prev_shield_pos = prev_debug_info.get('shield_pos', shield_pos)
            logger.info(f"  Shield pos: {shield_pos:.4f} (Δ{shield_pos-prev_shield_pos:+.4f})")
            
            logger.info(f"  Fan on: {debug_info.get('fan_on', False)}")
            
            # Store current debug info for next step delta calculation
            prev_debug_info = debug_info.copy()
        
        # Log new observation using schema
        obs_keys = getattr(unwrapped_env, 'OBS_KEYS', None)
        if obs_keys and len(obs) == len(obs_keys):
            # Use schema-based decoding
            new_obs_dict = {key: float(obs[i]) for i, key in enumerate(obs_keys)}
            next_obs_dict = new_obs_dict.copy()  # Store next observation (after action)
            logger.info("New Observation:")
            if 'plant_biomass_fraction' in new_obs_dict:
                curr_val = curr_obs_dict.get('plant_biomass_fraction', new_obs_dict['plant_biomass_fraction'])
                logger.info(f"  Plant biomass: {new_obs_dict['plant_biomass_fraction']:.4f} (Δ{new_obs_dict['plant_biomass_fraction']-curr_val:+.4f})")
            if 'plant_moisture' in new_obs_dict:
                new_moist = new_obs_dict['plant_moisture']
                curr_moist = curr_obs_dict.get('plant_moisture', new_moist)
                logger.info(f"  Moisture: {new_moist:.4f} (Δ{new_moist-curr_moist:+.4f})")
            if 'plant_nutrient' in new_obs_dict:
                new_nut = new_obs_dict['plant_nutrient']
                curr_nut = curr_obs_dict.get('plant_nutrient', new_nut)
                logger.info(f"  Nutrient: {new_nut:.4f} (Δ{new_nut-curr_nut:+.4f})")
            # Temperature and CO2 deltas are logged from debug info above
            if 'hw_shield_pos' in new_obs_dict:
                new_shield = new_obs_dict['hw_shield_pos']
                curr_shield = curr_obs_dict.get('hw_shield_pos', new_shield)
                logger.info(f"  Shield position: {new_shield:.4f} (Δ{new_shield-curr_shield:+.4f})")
            
            # Update observation tracking: shift for next iteration
            prev_obs_dict = curr_obs_dict.copy()  # Previous becomes current
            curr_obs_dict = next_obs_dict.copy()  # Current becomes next
        else:
            # Fallback for compatibility
            logger.info(f"New Observation (raw, len={len(obs)}):")
            for i in range(min(len(obs), 5)):
                logger.info(f"  obs[{i}]: {obs[i]:.4f}")
        
        if t % 6 == 0:
            unwrapped_env.render()
        
        if terminated or truncated:
            logger.warning(f"Episode ended: terminated={terminated}, truncated={truncated}")
            break
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("SIMULATION RUN COMPLETED")
    logger.info("="*80)
    logger.info(f"Total steps executed: {t+1}")
    logger.info(f"Final hour: {unwrapped_env.hour}")
    logger.info(f"Final step_count: {unwrapped_env.step_count}")
    logger.info(f"Total reward: {total_reward:.4f}")
    logger.info(f"Average reward per step: {total_reward/(t+1):.4f}")
    logger.info("\nFinal Plant State:")
    plant_state = unwrapped_env.plant.get_state()
    logger.info(f"  Canopy: {plant_state.get('canopy', 0):.4f}")
    logger.info(f"  Moisture: {plant_state.get('moisture', 0):.4f}")
    logger.info(f"  Nutrient: {plant_state.get('nutrient', 0):.4f}")
    logger.info(f"  Final LAI: {plant_state.get('LAI', 0):.4f}")
    total_biomass = np.sum(unwrapped_env.plant.organs.B_leaf) + np.sum(unwrapped_env.plant.organs.B_root) + np.sum(unwrapped_env.plant.organs.B_stem)
    logger.info(f"  Final total biomass: {total_biomass:.4f} g")
    logger.info(f"  Final leaf biomass: {np.sum(unwrapped_env.plant.organs.B_leaf):.4f} g")
    logger.info(f"  Final root biomass: {np.sum(unwrapped_env.plant.organs.B_root):.4f} g")
    logger.info(f"  Final stem biomass: {np.sum(unwrapped_env.plant.organs.B_stem):.4f} g")
    logger.info(f"  Water stress: {plant_state.get('stress_water', 1.0):.4f}")
    logger.info(f"  Temperature stress: {plant_state.get('stress_temp', 1.0):.4f}")
    logger.info(f"  Nutrient stress: {plant_state.get('stress_nutrient', 1.0):.4f}")
    logger.info("\nFinal Environment State:")
    logger.info(f"  Temperature: {unwrapped_env.env.T_middle:.2f}°C")
    logger.info(f"  Relative Humidity: {unwrapped_env.env.RH_middle:.2f}%")
    logger.info(f"\nFinal Hardware State:")
    logger.info(f"  Shield position: {unwrapped_env.hw.shield_pos:.4f}")
    logger.info(f"  Fan on: {unwrapped_env.hw.fan_on}")
    logger.info(f"  Total energy consumed: {unwrapped_env.hw.energy:.4f}")
    logger.info(f"\nLog file saved to: {log_file}")
    logger.info("="*80)
    
    print(f"[main] Sim finished. Total reward: {total_reward:.4f}")
    print(f"[main] Detailed log saved to: {log_file}")
    if model:
        print(f"[main] Used trained PPO model: {model_path}")


def sample_visual(args):
    out = args.out or "sample_visual.png"
    # create a single synthetic image and optionally composite with provided real image
    tmp_dir = ROOT / "tmp_sample"
    tmp_dir.mkdir(exist_ok=True)
    synth_path = tmp_dir / "synth_sample.png"
    print("[main] Generating one synthetic sample...")
    synth_generator.gen_dataset(out_dir=str(tmp_dir), n=1)
    synth_img = tmp_dir / "synth_0000.png"
    # if a real image exists, create a simple composite
    real_path = args.real or DEFAULT_REAL_IMAGE
    if os.path.exists(real_path):
        try:
            from PIL import Image
            base = Image.open(real_path).convert("RGBA").resize((224,224))
            overlay = Image.open(str(synth_img)).convert("RGBA").resize((224,224))
            # composite: overlay canopy with 50% alpha
            blended = Image.blend(base, overlay, alpha=0.55)
            blended.save(out)
            print(f"[main] Composite saved to {out} (using real image {real_path})")
            return
        except Exception as e:
            print("[main] Composite failed:", e)
    # otherwise just copy the synthetic image
    os.replace(str(synth_img), out)
    print(f"[main] Sample synthetic image saved to {out}")

def parse_args():
    p = argparse.ArgumentParser(description="Digital Twin - main orchestrator")
    sub = p.add_subparsers(dest="cmd")

    g = sub.add_parser("gen_synth", help="Generate synthetic images")
    g.add_argument("--out", type=str, help="output directory")
    g.add_argument("--n", type=int, help="number of images")

    d = sub.add_parser("demo_ppo", help="Run quick PPO demo training")

    tc = sub.add_parser("train_classifier", help="Train classifier on dataset")
    tc.add_argument("--data", type=str, help="data directory (ImageFolder layout)")
    tc.add_argument("--epochs", type=int, help="epochs")

    s = sub.add_parser("sim_run", help="Run a simulation (uses trained PPO model if available, else naive policy)")
    s.add_argument("--steps", type=int, help="timesteps to run")
    s.add_argument("--species", type=str, default=None, help="Plant species (kale, broccoli, radish, etc.)")
    s.add_argument("--model", type=str, default=None, help="Path to trained PPO model (default: auto-detect best)")
    s.add_argument("--use_extended_obs", action='store_true', help="Use extended observations (if model was trained with them)")
    s.add_argument("--dose_nutrients", action='store_true', help="Apply nutrient dose at start (N=0.1, P=0.05, K=0.1)")
    s.add_argument("--dose_N", type=float, default=None, help="Nitrogen dose (0..1)")
    s.add_argument("--dose_P", type=float, default=None, help="Phosphorus dose (0..1)")
    s.add_argument("--dose_K", type=float, default=None, help="Potassium dose (0..1)")
    s.add_argument("--no_model", action='store_true', help="Do not use a trained model")
    s.add_argument("--full_episode", action='store_true', help="Run exactly env.max_steps (full episode)")

    v=sub.add_parser("analyze_sim_logs", help="Analyze simulation logs")
    v.add_argument("--log_file", type=str, default=None, help="path to log file")
    v.add_argument("--out_dir", type=str, default="viz_output", help="output directory")
    v.add_argument("--plot_dir", type=str, default="viz_output/plots", help="plot directory")
    v.add_argument("--csv_out", type=str, default="viz_output/parsed_sim.csv", help="csv output file")
    v.add_argument("--summary_out", type=str, default="viz_output/summary.txt", help="summary output file")
    v.add_argument("--dash_out", type=str, default="viz_output/dashboard.html", help="dashboard output file")

    sv = sub.add_parser("sample_visual", help="Generate sample visual and optionally composite with real image")
    sv.add_argument("--out", type=str, help="output filename")
    sv.add_argument("--real", type=str, help="path to real image to composite with")

    return p.parse_args()

def main():
    args = parse_args()
    if not args.cmd:
        print("No command given. Use -h to see options.")
        return
    if args.cmd == "gen_synth":
        gen_synth(args)
    elif args.cmd == "demo_ppo":
        demo_ppo(args)
    elif args.cmd == "train_classifier":
        train_classifier_cmd(args)
    elif args.cmd == "sim_run":
        sim_run(args)
    elif args.cmd == "analyze_sim_logs":
        analyze_sim_logs(args.log_file)
    elif args.cmd == "sample_visual":
        sample_visual(args)
    else:
        print("Unknown command:", args.cmd)

if __name__ == "__main__":
    main()
