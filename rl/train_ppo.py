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
from rl.gym_env import ImprovedDigitalTwinEnv as DigitalTwinEnv
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
            diagnostics_info = {
                'episode_reward': episode_info.get('r', 0),
                'episode_length': episode_info.get('l', 0),
                'death_reason': episode_info.get('death_reason', None),
            }
            
            # Get cumulative stats from the current step's info
            if 'cumulative_growth' in info:
                diagnostics_info['cumulative_growth'] = info['cumulative_growth']
            if 'cumulative_water_used' in info:
                diagnostics_info['cumulative_water_used'] = info['cumulative_water_used']
            else:
                # Fallback: set to 0 if not available
                diagnostics_info['cumulative_water_used'] = 0
            
            # Log the episode
            try:
                self.diagnostics.log_episode(diagnostics_info, self.current_episode_actions)
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
        model.save('ppo_full')
        print("Training complete.")

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