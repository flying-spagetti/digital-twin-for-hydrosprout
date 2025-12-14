# rl/train_ppo.py
"""
Train PPO agent on the DigitalTwinEnv with PlantStructural model.
Tweaked for higher exploration to prevent "passive agent" failure.
"""

import argparse
import os
import sys
import torch.nn as nn
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from rl.gym_env import EnhancedDigitalTwinEnv as DigitalTwinEnv
from rl.wrappers import make_env
from rl.curriculum import CurriculumScheduler, CurriculumWrapper
from rl.diagnostics import TrainingDiagnostics
import numpy as np

# Try to load config if available
try:
    from config import load_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


class CurriculumCallback(BaseCallback):
    """Callback to update curriculum during training."""
    
    def __init__(self, curriculum: CurriculumScheduler, verbose=0):
        super().__init__(verbose)
        self.curriculum = curriculum
    
    def _on_step(self) -> bool:
        """Update curriculum based on current timesteps."""
        self.curriculum.update(self.num_timesteps)
        return True


class DiagnosticsCallback(BaseCallback):
    """Callback to track training diagnostics and generate visualizations."""
    
    def __init__(self, diagnostics: TrainingDiagnostics, log_freq: int = 100, verbose=0):
        super().__init__(verbose)
        self.diagnostics = diagnostics
        self.log_freq = log_freq  # Log plots every N episodes
        
        # Episode tracking
        self.current_episode_actions = []
        self.current_episode_info = None
    
    def _on_step(self) -> bool:
        """Log step-level data and track episode actions."""
        # Get info from the environment (stable-baselines3 structure)
        infos = self.locals.get('infos', [])
        if not infos:
            return True
        
        # Handle both single info and list of infos
        info = infos[0] if isinstance(infos, list) else infos
        
        # Track actions for this episode
        actions = self.locals.get('actions', None)
        if actions is not None:
            # Handle both single action and batch of actions
            if isinstance(actions, (list, np.ndarray)) and len(actions) > 0:
                action = actions[0] if isinstance(actions, (list, np.ndarray)) and len(actions.shape) > 1 else actions
                if isinstance(action, np.ndarray):
                    self.current_episode_actions.append(action.copy())
                else:
                    self.current_episode_actions.append(action)
        
        # Log step-level data if available
        if isinstance(info, dict) and 'plant' in info and 'env' in info:
            try:
                self.diagnostics.log_step(info['plant'], info['env'])
            except Exception as e:
                if self.verbose > 1:
                    print(f"[Diagnostics] Warning: Could not log step: {e}")
        
        # Check if episode ended (stable-baselines3 adds 'episode' key when done)
        if isinstance(info, dict) and 'episode' in info:
            # Episode completed - log it
            episode_info = info['episode']
            
            # Build info dict for diagnostics
            # FIXED: Include terminated/truncated flags for accurate death rate counting
            diagnostics_info = {
                'episode_reward': episode_info.get('r', 0),
                'episode_length': episode_info.get('l', 0),
                'death_reason': episode_info.get('death_reason', None),
                'terminated': episode_info.get('terminated', False),
                'truncated': episode_info.get('truncated', False),
            }
            
            # Get cumulative stats from the current step's info
            if 'cumulative_growth' in info:
                diagnostics_info['cumulative_growth'] = info['cumulative_growth']
            if 'cumulative_water' in info:
                diagnostics_info['cumulative_water_used'] = info['cumulative_water']
            elif 'cumulative_water_used' in info:
                diagnostics_info['cumulative_water_used'] = info['cumulative_water_used']
            else:
                # Fallback: set to 0 if not available
                diagnostics_info['cumulative_water_used'] = 0
            
            # Extract observation features from true_state (mean over episode)
            obs_features = {}
            if 'true_state' in info:
                true_state = info['true_state']
                
                # Plant features
                if 'plant' in true_state:
                    plant = true_state['plant']
                    obs_features['plant_biomass'] = plant.get('biomass', 0)
                    obs_features['plant_moisture'] = plant.get('moisture', 0)
                    obs_features['plant_nutrient'] = plant.get('nutrient', 0)
                    obs_features['plant_LAI'] = plant.get('LAI', 0)
                    obs_features['plant_stress_water'] = plant.get('stress_water', 0)
                    obs_features['plant_stress_temp'] = plant.get('stress_temp', 0)
                    obs_features['plant_stress_nutrient'] = plant.get('stress_nutrient', 0)
                    obs_features['plant_height'] = plant.get('height', 0)
                    obs_features['plant_NSC'] = plant.get('NSC', 0)
                    obs_features['plant_N_content'] = plant.get('N_content', 0)
                    obs_features['plant_total_biomass'] = plant.get('total_biomass', 0)
                
                # Environment features
                if 'env' in true_state:
                    env = true_state['env']
                    obs_features['env_T_top'] = env.get('T_top', 25.0)
                    obs_features['env_T_middle'] = env.get('T_middle', 25.0)
                    obs_features['env_T_bottom'] = env.get('T_bottom', 25.0)
                    obs_features['env_RH_top'] = env.get('RH_top', 60.0)
                    obs_features['env_RH_middle'] = env.get('RH_middle', 60.0)
                    obs_features['env_RH_bottom'] = env.get('RH_bottom', 60.0)
                    obs_features['env_CO2'] = env.get('CO2', 400.0)
                
                # Nutrient features
                if 'nutrients' in true_state:
                    nutrients = true_state['nutrients']
                    obs_features['nutrient_EC'] = nutrients.get('EC', 1.8)
                    obs_features['nutrient_pH'] = nutrients.get('pH', 6.0)
                    obs_features['nutrient_N_ppm'] = nutrients.get('N_ppm', 0)
                
                # Hardware features
                if 'hardware' in true_state:
                    hw = true_state['hardware']
                    obs_features['hw_shield_pos'] = hw.get('shield_pos', 0)
                    obs_features['hw_fan_on'] = 1.0 if hw.get('fan_on', False) else 0.0
                    obs_features['hw_moisture_std'] = hw.get('moisture_std', 0)
                    obs_features['hw_coverage_efficiency'] = hw.get('coverage_efficiency', 0)
                    obs_features['hw_water_efficiency'] = hw.get('water_efficiency', 0)
                
                # Peltier module states
                if 'peltier_states' in true_state:
                    peltier_states = true_state['peltier_states']
                    for i, peltier_power in enumerate(peltier_states):
                        obs_features[f'peltier_{i}'] = float(peltier_power)
            
            # Also try to compute temp_stress from T_middle
            if 'env_T_middle' in obs_features:
                T_mid = obs_features['env_T_middle']
                # Temperature stress: 1.0 = optimal (20-30°C), 0.0 = extreme
                if 20.0 <= T_mid <= 30.0:
                    obs_features['env_temp_stress'] = 1.0
                elif T_mid < 20.0:
                    obs_features['env_temp_stress'] = max(0.0, (T_mid - 10.0) / 10.0)
                else:
                    obs_features['env_temp_stress'] = max(0.0, 1.0 - (T_mid - 30.0) / 10.0)
            
            # Compute nutrient stress factors
            if 'nutrient_EC' in obs_features and 'nutrient_pH' in obs_features:
                EC = obs_features['nutrient_EC']
                pH = obs_features['nutrient_pH']
                # EC stress: optimal around 1.8
                EC_error = abs(EC - 1.8) / 1.8
                obs_features['nutrient_EC_stress'] = max(0.0, 1.0 - EC_error)
                # pH stress: optimal around 6.0
                pH_error = abs(pH - 6.0) / 2.0
                obs_features['nutrient_pH_stress'] = max(0.0, 1.0 - pH_error)
            
            # Log the episode with observation features
            try:
                self.diagnostics.log_episode(diagnostics_info, self.current_episode_actions, obs_features)
            except Exception as e:
                if self.verbose > 0:
                    print(f"[Diagnostics] Warning: Could not log episode: {e}")
            
            # Reset episode tracking
            self.current_episode_actions = []
            
            # Periodically generate plots
            num_episodes = len(self.diagnostics.episode_rewards)
            if num_episodes > 0 and num_episodes % self.log_freq == 0:
                if self.verbose > 0:
                    print(f"\n[Diagnostics] Generating plots at episode {num_episodes}...")
                try:
                    self.diagnostics.plot_training_progress()
                    self.diagnostics.save_summary()
                except Exception as e:
                    if self.verbose > 0:
                        print(f"[Diagnostics] Warning: Could not generate plots: {e}")
        
        return True
    
    def _on_training_end(self) -> None:
        """Generate final plots and summary when training ends."""
        if self.verbose > 0:
            print("\n[Diagnostics] Generating final training plots...")
        self.diagnostics.plot_training_progress()
        self.diagnostics.save_summary()

def create_env(cfg=None, use_extended_obs=False, include_soil_obs=False, 
               include_nutrient_actions=False, use_wrappers=True, 
               use_curriculum=False, curriculum=None):
    """Create and wrap environment for training."""
    if cfg is None:
        cfg = {}
    if use_extended_obs:
        cfg['use_extended_obs'] = True
    if include_soil_obs:
        cfg['include_soil_obs'] = True
    if include_nutrient_actions:
        cfg['include_nutrient_actions'] = True
    
    # 1. Create Base Env (or Wrapped Env)
    if use_wrappers:
        # make_env applies Clip, Normalize, ScaleReward, etc.
        env = make_env(cfg=cfg, use_wrappers=True, use_framestack=False)
    else:
        env = DigitalTwinEnv(cfg=cfg)
        env = Monitor(env)
    
    # 2. Apply Curriculum Wrapper (Outer Layer)
    if use_curriculum and curriculum is not None:
        env = CurriculumWrapper(env, curriculum)
        
    return env

def main(args):
    # Load config if provided
    cfg = {}
    if args.config and CONFIG_AVAILABLE:
        try:
            cfg = load_config(args.config)
            print(f"Loaded config from {args.config}")
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
    elif CONFIG_AVAILABLE:
        try:
            cfg = load_config()
        except:
            pass
    
    # Check for soil-related flags
    include_soil_obs = getattr(args, 'include_soil_obs', False)
    include_nutrient_actions = getattr(args, 'include_nutrient_actions', False)
    use_curriculum = getattr(args, 'use_curriculum', False)
    use_wrappers = getattr(args, 'use_wrappers', True)
    
    # Setup curriculum learning
    curriculum = None
    if use_curriculum:
        # Total timesteps is critical for calculating curriculum stages
        curriculum = CurriculumScheduler(total_timesteps=args.timesteps)
        print(f"Curriculum learning enabled: {len(curriculum.stages)} stages defined.")
        print(f"Initial Stage: {curriculum.current_stage}")
    
    # Create training environment
    env = create_env(
        cfg=cfg, 
        use_extended_obs=args.extended,
        include_soil_obs=include_soil_obs,
        include_nutrient_actions=include_nutrient_actions,
        use_wrappers=use_wrappers,
        use_curriculum=use_curriculum,
        curriculum=curriculum
    )
    
    # Create evaluation environment (No curriculum wrapper for consistent benchmarks)
    eval_env = create_env(
        cfg=cfg, 
        use_extended_obs=args.extended,
        include_soil_obs=include_soil_obs,
        include_nutrient_actions=include_nutrient_actions,
        use_wrappers=False,  # Raw env for true performance eval
        use_curriculum=False 
    )
    eval_env = Monitor(eval_env)
    
    # Setup diagnostics
    log_dir = args.log_dir if hasattr(args, 'log_dir') and args.log_dir else './training_logs'
    diagnostics = TrainingDiagnostics(log_dir=log_dir)
    
    # Setup callbacks
    callbacks = []
    
    if use_curriculum and curriculum is not None:
        callbacks.append(CurriculumCallback(curriculum))
    
    # Diagnostics callback (log plots every 100 episodes)
    diagnostics_callback = DiagnosticsCallback(
        diagnostics=diagnostics,
        log_freq=100,
        verbose=1
    )
    callbacks.append(diagnostics_callback)
    
    # Evaluate occasionally to check progress
    # EvalCallback saves best model (based on evaluation performance) to ./ppo_best/best_model.zip
    # It will replace the previous best model if a new one performs better
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./ppo_best/',
        log_path='./ppo_logs/',
        eval_freq=max(1000, args.timesteps // 20),
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(5000, args.timesteps // 10),
        save_path='./ppo_checkpoints/',
        name_prefix='ppo_model'
    )
    callbacks.append(checkpoint_callback)
    
    callback = CallbackList(callbacks) if len(callbacks) > 0 else None
    
    # --- PPO HYPERPARAMETERS (TUNED) ---
    def constant_lr_schedule(progress_remaining):
        # Increased to 3e-4 to speed up initial learning
        return 3e-4

    ppo_kwargs = {
        'policy': 'MlpPolicy',
        'env': env,
        'verbose': 1,
        'learning_rate': constant_lr_schedule,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.995,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        # FIXED: Proper entropy for control tasks (was 0.05, now 0.005) (Fan/Water) instead of doing nothing
        'ent_coef': 0.005,  # FIXED: Reduced from 0.05 (was 10x too high) 
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'tensorboard_log': './ppo_tensorboard/',
        'device': 'auto', # Use GPU if available
        'policy_kwargs': {
            'net_arch': dict(pi=[256, 256], vf=[256, 256]),
            'activation_fn': nn.Tanh,
        },
    }
    
    # Allow config file to override specific PPO params
    if cfg and 'ppo' in cfg and cfg['ppo']:
        ppo_config = cfg['ppo']
        # Remove LR from config if we want to enforce our schedule, 
        # or add logic here to use config's LR.
        # For now, we trust the hardcoded schedule for stability.
        if 'learning_rate' in ppo_config:
            del ppo_config['learning_rate']
        ppo_kwargs.update(ppo_config)
    
    # Train
    if args.demo:
        print("Running quick PPO demo training...")
        ppo_kwargs['n_steps'] = 512
        ppo_kwargs['batch_size'] = 32
        model = PPO(**ppo_kwargs)
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save('ppo_demo')
        print("Demo complete.")
    else:
        print(f"Starting full PPO training for {args.timesteps} timesteps...")
        model = PPO(**ppo_kwargs)
        model.learn(total_timesteps=args.timesteps, callback=callback)
        # Save final model (this will overwrite previous ppo_full.zip if it exists)
        model.save('ppo_full')
        print("Training complete.")
        print(f"\nSaved outputs:")
        print(f"  - Best model (by evaluation): ./ppo_best/best_model.zip")
        print(f"  - Final model: ./ppo_full.zip")
        print(f"  - Checkpoints: ./ppo_checkpoints/")
        print(f"  - Training logs: {log_dir}/")
        print(f"    * training_progress.png (comprehensive plots)")
        print(f"    * summary.json (training statistics)")
        print(f"  - Evaluation logs: ./ppo_logs/evaluations.npz")
        print(f"  - TensorBoard logs: ./ppo_tensorboard/ (view with: tensorboard --logdir ./ppo_tensorboard/)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PPO on DigitalTwin')
    parser.add_argument('--demo', action='store_true', help='Run quick 2k step demo')
    parser.add_argument('--extended', action='store_true', help='Use extended observations')
    parser.add_argument('--include_soil_obs', action='store_true', help='Include soil metrics')
    parser.add_argument('--include_nutrient_actions', action='store_true', help='Enable nutrient dosing')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--timesteps', type=int, default=None, help='Total training timesteps')
    parser.add_argument('--use_curriculum', action='store_true', help='Enable curriculum learning')
    parser.add_argument('--no_wrappers', action='store_true', help='Disable standard wrappers')
    parser.add_argument('--log_dir', type=str, default='./training_logs', help='Directory for training logs and diagnostics')
    
    args = parser.parse_args()
    
    # Default timesteps if not provided
    if args.timesteps is None:
        # 300k is a good baseline for this curriculum (allows ~30k steps per stage transition)
        args.timesteps = 2000 if args.demo else 300000 
    
    # Logic inversion for clarity
    args.use_wrappers = not args.no_wrappers
    
    main(args)