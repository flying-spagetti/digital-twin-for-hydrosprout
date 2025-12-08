# rl/train_ppo.py
"""
Train PPO agent on the DigitalTwinEnv with PlantStructural model.
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
from rl.gym_env import DigitalTwinEnv
from rl.wrappers import make_env
from rl.curriculum import CurriculumScheduler, CurriculumWrapper

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
    
    if use_wrappers:
        # Use our custom wrapper factory
        env = make_env(cfg=cfg, use_wrappers=True, use_framestack=False)
    else:
        # Raw env
        env = DigitalTwinEnv(cfg=cfg)
        env = Monitor(env)
    
    # Apply curriculum wrapper if requested
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
    
    # Setup curriculum learning if requested
    curriculum = None
    if use_curriculum:
        # Ensure total_timesteps is set for curriculum scaling
        curriculum = CurriculumScheduler(total_timesteps=args.timesteps)
        print(f"Curriculum learning enabled with {len(curriculum.stages)} stages")
    
    # Create environment
    env = create_env(
        cfg=cfg, 
        use_extended_obs=args.extended,
        include_soil_obs=include_soil_obs,
        include_nutrient_actions=include_nutrient_actions,
        use_wrappers=use_wrappers,
        use_curriculum=use_curriculum,
        curriculum=curriculum
    )
    
    # Create evaluation environment (no curriculum for consistent metric comparison)
    eval_env = create_env(
        cfg=cfg, 
        use_extended_obs=args.extended,
        include_soil_obs=include_soil_obs,
        include_nutrient_actions=include_nutrient_actions,
        use_wrappers=False,  # No wrappers for evaluation usually
        use_curriculum=False 
    )
    eval_env = Monitor(eval_env)
    
    # Setup callbacks
    callbacks = []
    
    # Curriculum callback
    if use_curriculum and curriculum is not None:
        callbacks.append(CurriculumCallback(curriculum))
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./ppo_best/',
        log_path='./ppo_logs/',
        eval_freq=max(1000, args.timesteps // 20),
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(5000, args.timesteps // 10),
        save_path='./ppo_checkpoints/',
        name_prefix='ppo_model'
    )
    callbacks.append(checkpoint_callback)
    
    callback = CallbackList(callbacks) if len(callbacks) > 0 else None
    
    # Hyperparameters - Tuned for stability
    def constant_lr_schedule(progress_remaining):
        return 1e-4

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
        'ent_coef': 0.02, # Enforce exploration
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'tensorboard_log': './ppo_tensorboard/',
        'policy_kwargs': {
            'net_arch': dict(pi=[256, 256], vf=[256, 256]),
            'activation_fn': nn.Tanh,
        },
    }
    
    # Override with config if available
    if cfg and 'ppo' in cfg and cfg['ppo']:
        ppo_config = cfg['ppo'].copy() if isinstance(cfg['ppo'], dict) else {}
        
        # Handle learning_rate separately - must be callable
        if 'learning_rate' in ppo_config:
            lr_val = float(ppo_config['learning_rate'])
            def config_lr_schedule(progress_remaining):
                return lr_val
            ppo_kwargs['learning_rate'] = config_lr_schedule
            del ppo_config['learning_rate']  # Remove to avoid overwrite
        
        # Update other parameters
        ppo_kwargs.update(ppo_config)
    
    # Final safety check: ensure learning_rate is always callable
    if not callable(ppo_kwargs.get('learning_rate')):
        lr_val = float(ppo_kwargs.get('learning_rate', 1e-4))
        def final_lr_schedule(progress_remaining):
            return lr_val
        ppo_kwargs['learning_rate'] = final_lr_schedule
    
    # Train
    if args.demo:
        print("Running quick PPO demo...")
        ppo_kwargs['n_steps'] = 512
        ppo_kwargs['batch_size'] = 32
        model = PPO(**ppo_kwargs)
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save('ppo_demo')
    else:
        print("Running full PPO training...")
        model = PPO(**ppo_kwargs)
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save('ppo_full')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run quick demo')
    parser.add_argument('--extended', action='store_true', help='Use extended observations')
    parser.add_argument('--include_soil_obs', action='store_true', help='Include soil metrics')
    parser.add_argument('--include_nutrient_actions', action='store_true', help='Include nutrient actions')
    parser.add_argument('--config', type=str, default=None, help='Path to config')
    parser.add_argument('--timesteps', type=int, default=None, help='Training timesteps')
    parser.add_argument('--use_curriculum', action='store_true', help='Enable curriculum')
    parser.add_argument('--no_wrappers', action='store_true', help='Disable wrappers')
    args = parser.parse_args()
    
    if args.timesteps is None:
        args.timesteps = 2000 if args.demo else 300000 # Default increased to support curriculum
    
    args.use_wrappers = not args.no_wrappers
    
    main(args)