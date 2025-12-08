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
from rl.gym_env import DigitalTwinEnv

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
    steps = args.steps or 48
    
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
    
    env = DigitalTwinEnv()
    
    # Apply initial nutrient dose if provided
    if nutrient_dose and hasattr(env.plant, 'plant') and hasattr(env.plant.plant, 'soil'):
        env.plant.plant.soil.add_dose(
            N=nutrient_dose['N'],
            P=nutrient_dose['P'],
            K=nutrient_dose['K'],
            micro=nutrient_dose.get('micro'),
            chelated=nutrient_dose.get('chelated', False)
        )
        print(f"[main] Applied initial nutrient dose: N={nutrient_dose['N']:.3f}, P={nutrient_dose['P']:.3f}, K={nutrient_dose['K']:.3f}")
    
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
                
                # Create environment with matching observation space
                env_cfg = {}
                if use_extended:
                    env_cfg['use_extended_obs'] = True
                
                # Recreate env with correct config
                env = DigitalTwinEnv(cfg=env_cfg)
                
                # Re-apply nutrient dose if env was recreated
                if nutrient_dose and hasattr(env.plant, 'plant') and hasattr(env.plant.plant, 'soil'):
                    env.plant.plant.soil.add_dose(
                        N=nutrient_dose['N'],
                        P=nutrient_dose['P'],
                        K=nutrient_dose['K'],
                        micro=nutrient_dose.get('micro'),
                        chelated=nutrient_dose.get('chelated', False)
                    )
                
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
    
    # Reset environment
    obs, info = env.reset()
    logger.info("Environment reset")
    logger.info(f"Initial hour: {env.hour}")
    logger.info(f"Initial step_count: {env.step_count}")
    logger.info(f"Max steps: {env.max_steps}")
    
    # Log initial observation
    if len(obs) >= 7:
        canopy, moist, nut, pmold, tmp_scaled, lux, shield_pos = obs[:7]
    else:
        canopy, moist, nut, pmold, tmp_scaled, lux, shield_pos = obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6] if len(obs) > 6 else 0.0
    temp = tmp_scaled * 40.0
    logger.info("Initial Observation:")
    logger.info(f"  Canopy: {canopy:.4f}")
    logger.info(f"  Moisture: {moist:.4f}")
    logger.info(f"  Nutrient: {nut:.4f}")
    logger.info(f"  Mold probability: {pmold:.4f}")
    logger.info(f"  Temperature (scaled): {tmp_scaled:.4f} -> {temp:.2f}°C")
    logger.info(f"  Lux: {lux:.4f}")
    logger.info(f"  Shield position: {shield_pos:.4f}")
    
    # Log initial plant state
    logger.info("Initial Plant State:")
    logger.info(f"  Canopy (C): {env.plant.C:.4f}")
    logger.info(f"  Moisture (M): {env.plant.M:.4f}")
    logger.info(f"  Nutrient (N): {env.plant.N:.4f}")
    logger.info(f"  Mold probability (P_mold): {env.plant.P_mold:.4f}")
    # PlantStructural-specific details
    if hasattr(env.plant, 'plant') and hasattr(env.plant.plant, 'LAI'):
        logger.info(f"  LAI (Leaf Area Index): {env.plant.plant.LAI:.4f}")
        logger.info(f"  Total leaf biomass: {np.sum(env.plant.plant.B_leaf):.4f} g")
        logger.info(f"  Total root biomass: {np.sum(env.plant.plant.B_root):.4f} g")
        logger.info(f"  Total stem biomass: {np.sum(env.plant.plant.B_stem):.4f} g")
        logger.info(f"  Number of plants: {env.plant.plant.n}")
        
        # Log soil status if SoilModelExtended is available
        if hasattr(env.plant.plant, 'soil'):
            soil = env.plant.plant.soil
            soil_status = soil.status()
            logger.info("Initial Soil Status:")
            logger.info(f"  pH: {soil_status.get('pH', 0):.2f}")
            logger.info(f"  EC proxy: {soil_status.get('ec', 0):.3f}")
            logger.info(f"  Macronutrients:")
            logger.info(f"    N: {soil_status.get('soil_N', 0):.4f}")
            logger.info(f"    P: {soil_status.get('soil_P', 0):.4f}")
            logger.info(f"    K: {soil_status.get('soil_K', 0):.4f}")
            logger.info(f"  Micronutrients:")
            for micro in ['Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']:
                if f'soil_{micro}' in soil_status:
                    logger.info(f"    {micro}: {soil_status[f'soil_{micro}']:.4f}")
    
    # Log initial environment state
    logger.info("Initial Environment State:")
    logger.info(f"  Temperature: {env.env.T:.2f}°C")
    logger.info(f"  Relative Humidity: {env.env.RH:.2f}%")
    logger.info(f"  Light: {env.env.solar_input(env.hour):.4f}")
    
    # Log initial hardware state
    logger.info("Initial Hardware State:")
    logger.info(f"  Shield position: {env.hw.shield_pos:.4f}")
    logger.info(f"  Fan on: {env.hw.fan_on}")
    logger.info(f"  Energy: {env.hw.energy:.4f}")
    
    logger.info("-"*80)
    
    total_reward = 0.0
    action_history = []
    
    for t in range(steps):
        logger.info(f"\n{'='*80}")
        logger.info(f"STEP {t+1}/{steps} | Hour {env.hour}")
        logger.info(f"{'='*80}")
        
        # Log current observation
        if len(obs) >= 7:
            canopy, moist, nut, pmold, tmp_scaled, lux, shield_pos = obs[:7]
        else:
            canopy, moist, nut, pmold, tmp_scaled, lux, shield_pos = obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6] if len(obs) > 6 else 0.0
        temp = tmp_scaled * 40.0
        logger.info("Current Observation:")
        logger.info(f"  Canopy: {canopy:.4f}")
        logger.info(f"  Moisture: {moist:.4f}")
        logger.info(f"  Nutrient: {nut:.4f}")
        logger.info(f"  Mold probability: {pmold:.4f}")
        logger.info(f"  Temperature (scaled): {tmp_scaled:.4f} -> {temp:.2f}°C")
        logger.info(f"  Lux: {lux:.4f}")
        logger.info(f"  Shield position: {shield_pos:.4f}")
        
        # Determine action - use trained model if available, else naive policy
        if model:
            # Use trained PPO model
            action, _ = model.predict(obs, deterministic=True)
            action = action.tolist() if hasattr(action, 'tolist') else list(action)
            action_reasons = [f"PPO model decision (deterministic)"]
            logger.info("Using trained PPO model for action selection")
        else:
            # Fallback to naive policy
            action = [0.0, 0.0, 0.0, 0.0]  # water, fan, shield_delta, heater
            action_reasons = []
            
            if moist < 0.35:
                action[0] = 0.8  # water
                action_reasons.append(f"Low moisture ({moist:.3f} < 0.35)")
            if temp > 28.0:
                action[1] = 1.0  # fan on
                action[2] = 0.2  # open shield slightly
                action_reasons.append(f"High temperature ({temp:.2f}°C > 28°C)")
            if temp < 16.0:
                action[3] = 0.6  # heater
                action_reasons.append(f"Low temperature ({temp:.2f}°C < 16°C)")
            
            if not action_reasons:
                action_reasons.append("No action needed (all conditions normal)")
        
        # Log action
        logger.info("Action Selected:")
        logger.info(f"  Water: {action[0]:.2f}")
        logger.info(f"  Fan: {action[1]:.2f} ({'ON' if action[1] > 0.5 else 'OFF'})")
        logger.info(f"  Shield delta: {action[2]:.2f}")
        logger.info(f"  Heater: {action[3]:.2f}")
        logger.info(f"  Reasons: {', '.join(action_reasons)}")
        action_history.append(action.copy())
        
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
        logger.info("Plant State After Step:")
        logger.info(f"  Canopy (C): {env.plant.C:.4f}")
        logger.info(f"  Moisture (M): {env.plant.M:.4f}")
        logger.info(f"  Nutrient (N): {env.plant.N:.4f}")
        logger.info(f"  Mold probability (P_mold): {env.plant.P_mold:.4f}")
        # PlantStructural-specific details
        if hasattr(env.plant, 'plant') and hasattr(env.plant.plant, 'LAI'):
            logger.info(f"  LAI (Leaf Area Index): {env.plant.plant.LAI:.4f}")
            logger.info(f"  Total leaf biomass: {np.sum(env.plant.plant.B_leaf):.4f} g")
            logger.info(f"  Total root biomass: {np.sum(env.plant.plant.B_root):.4f} g")
            logger.info(f"  Total stem biomass: {np.sum(env.plant.plant.B_stem):.4f} g")
            diag = plant_info.get('diagnostics', {})
            if diag:
                logger.info(f"  Transpiration: {diag.get('transp_total_liters', 0):.4f} L/h")
                logger.info(f"  VPD (normalized): {diag.get('vpd_norm', 0):.4f}")
            
            # Log soil status after step
            if hasattr(env.plant.plant, 'soil'):
                soil = env.plant.plant.soil
                soil_status = soil.status()
                logger.info("Soil Status After Step:")
                logger.info(f"  pH: {soil_status.get('pH', 0):.2f}")
                logger.info(f"  EC proxy: {soil_status.get('ec', 0):.3f}")
                logger.info(f"  Macronutrients: N={soil_status.get('soil_N', 0):.4f}, P={soil_status.get('soil_P', 0):.4f}, K={soil_status.get('soil_K', 0):.4f}")
                # Check for toxicity warnings
                tox_warnings = soil.toxicity_warnings()
                if tox_warnings:
                    logger.warning(f"  Toxicity warnings: {tox_warnings}")
        if plant_info:
            logger.info(f"  Plant info: {plant_info}")
        
        # Log environment state from info
        env_info = info.get('env', {})
        logger.info("Environment State After Step:")
        logger.info(f"  Temperature: {env.env.T:.2f}°C")
        logger.info(f"  Relative Humidity: {env_info.get('RH', env.env.RH):.2f}%")
        logger.info(f"  Light (L): {env_info.get('L', 0):.4f}")
        logger.info(f"  Evaporation: {env_info.get('evap', 0):.4f}")
        if env_info:
            logger.info(f"  Env info: {env_info}")
        
        # Log hardware state from info
        hw_info = info.get('hw', {})
        logger.info("Hardware State After Step:")
        logger.info(f"  Shield position: {env.hw.shield_pos:.4f}")
        logger.info(f"  Fan on: {env.hw.fan_on}")
        logger.info(f"  Energy consumed: {env.hw.energy:.4f}")
        logger.info(f"  Delivered water: {hw_info.get('delivered_water', 0):.4f}")
        logger.info(f"  Heater power: {hw_info.get('heater_power', 0):.4f}")
        if hw_info:
            logger.info(f"  HW info: {hw_info}")
        
        # Log new observation
        if len(obs) >= 7:
            new_canopy, new_moist, new_nut, new_pmold, new_tmp_scaled, new_lux, new_shield_pos = obs[:7]
        else:
            new_canopy, new_moist, new_nut, new_pmold, new_tmp_scaled, new_lux, new_shield_pos = obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6] if len(obs) > 6 else 0.0
        new_temp = new_tmp_scaled * 40.0
        logger.info("New Observation:")
        logger.info(f"  Canopy: {new_canopy:.4f} (Δ{new_canopy-canopy:+.4f})")
        logger.info(f"  Moisture: {new_moist:.4f} (Δ{new_moist-moist:+.4f})")
        logger.info(f"  Nutrient: {new_nut:.4f} (Δ{new_nut-nut:+.4f})")
        logger.info(f"  Mold probability: {new_pmold:.4f} (Δ{new_pmold-pmold:+.4f})")
        logger.info(f"  Temperature: {new_temp:.2f}°C (Δ{new_temp-temp:+.2f}°C)")
        logger.info(f"  Lux: {new_lux:.4f} (Δ{new_lux-lux:+.4f})")
        logger.info(f"  Shield position: {new_shield_pos:.4f} (Δ{new_shield_pos-shield_pos:+.4f})")
        
        if t % 6 == 0:
            env.render()
        
        if terminated or truncated:
            logger.warning(f"Episode ended: terminated={terminated}, truncated={truncated}")
            break
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("SIMULATION RUN COMPLETED")
    logger.info("="*80)
    logger.info(f"Total steps executed: {t+1}")
    logger.info(f"Final hour: {env.hour}")
    logger.info(f"Final step_count: {env.step_count}")
    logger.info(f"Total reward: {total_reward:.4f}")
    logger.info(f"Average reward per step: {total_reward/(t+1):.4f}")
    logger.info("\nFinal Plant State:")
    logger.info(f"  Canopy (C): {env.plant.C:.4f}")
    logger.info(f"  Moisture (M): {env.plant.M:.4f}")
    logger.info(f"  Nutrient (N): {env.plant.N:.4f}")
    logger.info(f"  Mold probability (P_mold): {env.plant.P_mold:.4f}")
    # PlantStructural-specific final details
    if hasattr(env.plant, 'plant') and hasattr(env.plant.plant, 'LAI'):
        logger.info(f"  Final LAI: {env.plant.plant.LAI:.4f}")
        logger.info(f"  Final total biomass: {np.sum(env.plant.plant.B_leaf + env.plant.plant.B_root + env.plant.plant.B_stem):.4f} g")
        logger.info(f"  Final leaf biomass: {np.sum(env.plant.plant.B_leaf):.4f} g")
        logger.info(f"  Final root biomass: {np.sum(env.plant.plant.B_root):.4f} g")
        logger.info(f"  Final stem biomass: {np.sum(env.plant.plant.B_stem):.4f} g")
        
        # Final soil status
        if hasattr(env.plant.plant, 'soil'):
            soil = env.plant.plant.soil
            soil_status = soil.status()
            logger.info("\nFinal Soil Status:")
            logger.info(f"  pH: {soil_status.get('pH', 0):.2f}")
            logger.info(f"  EC proxy: {soil_status.get('ec', 0):.3f}")
            logger.info(f"  Macronutrients:")
            logger.info(f"    N: {soil_status.get('soil_N', 0):.4f}")
            logger.info(f"    P: {soil_status.get('soil_P', 0):.4f}")
            logger.info(f"    K: {soil_status.get('soil_K', 0):.4f}")
            logger.info(f"  Micronutrients:")
            for micro in ['Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo', 'Se', 'Cl']:
                if f'soil_{micro}' in soil_status:
                    logger.info(f"    {micro}: {soil_status[f'soil_{micro}']:.4f}")
            # Final toxicity check
            tox_warnings = soil.toxicity_warnings()
            if tox_warnings:
                logger.warning(f"  Final toxicity warnings: {tox_warnings}")
    logger.info("\nFinal Environment State:")
    logger.info(f"  Temperature: {env.env.T:.2f}°C")
    logger.info(f"  Relative Humidity: {env.env.RH:.2f}%")
    logger.info(f"\nFinal Hardware State:")
    logger.info(f"  Shield position: {env.hw.shield_pos:.4f}")
    logger.info(f"  Fan on: {env.hw.fan_on}")
    logger.info(f"  Total energy consumed: {env.hw.energy:.4f}")
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
    s.add_argument("--model", type=str, default=None, help="Path to trained PPO model (default: auto-detect best)")
    s.add_argument("--use_extended_obs", action='store_true', help="Use extended observations (if model was trained with them)")
    s.add_argument("--dose_nutrients", action='store_true', help="Apply nutrient dose at start (N=0.1, P=0.05, K=0.1)")
    s.add_argument("--dose_N", type=float, default=None, help="Nitrogen dose (0..1)")
    s.add_argument("--dose_P", type=float, default=None, help="Phosphorus dose (0..1)")
    s.add_argument("--dose_K", type=float, default=None, help="Potassium dose (0..1)")
    s.add_argument("--no_model", action='store_true', help="Do not use a trained model")

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
