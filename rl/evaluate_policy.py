#!/usr/bin/env python3
"""
evaluate_policy.py

Deterministic evaluation harness for RL policy.
Runs N episodes with fixed seed and prints comprehensive statistics.

Usage:
    python rl/evaluate_policy.py --model ppo_best/best_model.zip --n_episodes 20
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict
from stable_baselines3 import PPO

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rl.gym_env import EnhancedDigitalTwinEnv as DigitalTwinEnv
from rl.wrappers import make_env


def get_unwrapped_env(env):
    """Helper to get unwrapped env for attribute access."""
    unwrapped = env
    while hasattr(unwrapped, 'env'):
        next_env = unwrapped.env
        if hasattr(next_env, 'hour') and hasattr(next_env, 'plant') and hasattr(next_env, 'hw'):
            unwrapped = next_env
            break
        if not (hasattr(next_env, 'reset') and hasattr(next_env, 'step')):
            break
        unwrapped = next_env
    return unwrapped


def evaluate_policy(model_path: str, n_episodes: int = 20, seed: int = 42, 
                   deterministic: bool = True, verbose: bool = True):
    """
    Evaluate a trained policy with fixed seed for reproducibility.
    
    Args:
        model_path: Path to trained PPO model
        n_episodes: Number of episodes to run
        seed: Random seed for reproducibility
        deterministic: Use deterministic policy (True) or stochastic (False)
        verbose: Print detailed output
    
    Returns:
        Dict with evaluation statistics
    """
    # Load model
    if verbose:
        print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Create environment (no wrappers for true evaluation)
    env = DigitalTwinEnv()
    # Note: make_env applies wrappers, so we use raw env for evaluation
    # env = make_env(use_wrappers=False, use_framestack=False)  # Not needed for raw env
    
    # Evaluation statistics
    episode_rewards = []
    episode_lengths = []
    episode_biomass = []
    episode_stresses = defaultdict(list)
    termination_reasons = []
    max_steps_per_episode = []
    
    # New metrics requested
    episodes_reaching_max_steps = 0
    daytime_photosynthesis = []  # Photosynthesis during hours 6-18
    water_action_high_vpd = []  # Water action during high VPD/high temp windows
    water_action_high_temp = []  # Water action during high temp windows
    
    # Detailed trajectory for one episode
    detailed_trajectory = []
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"EVALUATION: {n_episodes} episodes, seed={seed}, deterministic={deterministic}")
        print(f"{'='*80}\n")
    
    # Get unwrapped env once for hour access
    unwrapped_env_eval = get_unwrapped_env(env)
    
    for episode in range(n_episodes):
        # Set seed for reproducibility
        np.random.seed(seed + episode)
        obs, info = env.reset(seed=seed + episode)
        
        episode_reward = 0.0
        episode_length = 0
        episode_biomass_gain = 0.0
        episode_stress_water = []
        episode_stress_temp = []
        episode_stress_nutrient = []
        terminated = False
        truncated = False
        
        # Get max_steps from info or env
        max_steps = info.get('max_steps', getattr(unwrapped_env_eval, 'max_steps', 336))
        max_steps_per_episode.append(max_steps)
        
        # Track metrics for this episode
        episode_daytime_photosynthesis = []
        episode_water_high_vpd = []
        episode_water_high_temp = []
        
        # Collect detailed trajectory for first episode only
        collect_trajectory = (episode == 0)
        
        while not (terminated or truncated):
            # Get action from policy
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Collect state information
            if 'true_state' in info:
                true_state = info['true_state']
                if 'plant' in true_state:
                    plant = true_state['plant']
                    episode_stress_water.append(plant.get('stress_water', 1.0))
                    episode_stress_temp.append(plant.get('stress_temp', 1.0))
                    episode_stress_nutrient.append(plant.get('stress_nutrient', 1.0))
                    
                    # Track biomass gain
                    if episode_length == 1:
                        initial_biomass = plant.get('total_biomass', 0.0)
                    else:
                        current_biomass = plant.get('total_biomass', 0.0)
                        episode_biomass_gain = current_biomass - initial_biomass
            
            # Track new metrics
            # Get hour from info or unwrapped env
            current_hour = info.get('hour', getattr(unwrapped_env_eval, 'hour', 12))
            
            # Daytime photosynthesis (hours 6-18)
            if 6 <= current_hour <= 18:
                if 'plant' in info:
                    photo_rate = info['plant'].get('photosynthesis_rate', 0.0)
                    episode_daytime_photosynthesis.append(photo_rate)
            
            # Water action during high VPD / high temp windows
            if 'debug' in info:
                debug = info['debug']
                T_middle = debug.get('T_middle', 25.0)
                # High temp: > 28°C
                if T_middle > 28.0:
                    if 'applied_action' in info:
                        water_action = info['applied_action'].get('water_total', 0.0)
                        episode_water_high_temp.append(water_action)
                
                # High VPD: estimated from T and RH (simplified: T > 25°C and RH < 60%)
                # VPD increases with temperature and decreases with RH
                RH = debug.get('RH', 60.0)
                if T_middle > 25.0 and RH < 60.0:
                    if 'applied_action' in info:
                        water_action = info['applied_action'].get('water_total', 0.0)
                        episode_water_high_vpd.append(water_action)
            
            # Collect detailed trajectory for first episode
            if collect_trajectory and episode_length <= 48:  # First 48 steps
                step_data = {
                    'step': episode_length,
                    'reward': float(reward),
                    'terminated': terminated,
                    'truncated': truncated,
                }
                if 'true_state' in info:
                    step_data['true_state'] = info['true_state'].copy()
                if 'observed_state' in info:
                    step_data['observed_state'] = info['observed_state'].copy()
                if 'applied_action' in info:
                    step_data['applied_action'] = info['applied_action'].copy()
                detailed_trajectory.append(step_data)
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_biomass.append(episode_biomass_gain)
        
        # Check if episode reached max steps
        if episode_length >= max_steps:
            episodes_reaching_max_steps += 1
        
        if episode_stress_water:
            episode_stresses['water'].append(np.mean(episode_stress_water))
        if episode_stress_temp:
            episode_stresses['temp'].append(np.mean(episode_stress_temp))
        if episode_stress_nutrient:
            episode_stresses['nutrient'].append(np.mean(episode_stress_nutrient))
        
        # Record new metrics
        if episode_daytime_photosynthesis:
            daytime_photosynthesis.append(np.mean(episode_daytime_photosynthesis))
        if episode_water_high_vpd:
            water_action_high_vpd.append(np.mean(episode_water_high_vpd))
        if episode_water_high_temp:
            water_action_high_temp.append(np.mean(episode_water_high_temp))
        
        # Record termination reason
        death_reason = info.get('death_reason', None)
        if terminated:
            termination_reasons.append(death_reason if death_reason else 'terminated')
        elif truncated:
            termination_reasons.append('truncated')
        else:
            termination_reasons.append('completed')
        
        if verbose and (episode + 1) % 5 == 0:
            print(f"Episode {episode + 1}/{n_episodes}: "
                  f"reward={episode_reward:.2f}, length={episode_length}, "
                  f"reason={termination_reasons[-1]}")
    
    # Compute statistics
    pct_reaching_max_steps = (episodes_reaching_max_steps / n_episodes) * 100
    avg_daytime_photosynthesis = np.mean(daytime_photosynthesis) if daytime_photosynthesis else 0.0
    avg_water_high_vpd = np.mean(water_action_high_vpd) if water_action_high_vpd else 0.0
    avg_water_high_temp = np.mean(water_action_high_temp) if water_action_high_temp else 0.0
    
    stats = {
        'n_episodes': n_episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'mean_biomass': np.mean(episode_biomass),
        'std_biomass': np.std(episode_biomass),
        'mean_stress_water': np.mean(episode_stresses.get('water', [1.0])),
        'mean_stress_temp': np.mean(episode_stresses.get('temp', [1.0])),
        'mean_stress_nutrient': np.mean(episode_stresses.get('nutrient', [1.0])),
        'pct_reaching_max_steps': pct_reaching_max_steps,
        'termination_reasons': dict(zip(*np.unique(termination_reasons, return_counts=True))),
        'detailed_trajectory': detailed_trajectory,
        # New metrics
        'pct_reaching_336_steps': pct_reaching_max_steps,
        'avg_daytime_photosynthesis': avg_daytime_photosynthesis,
        'avg_water_action_high_vpd': avg_water_high_vpd,
        'avg_water_action_high_temp': avg_water_high_temp,
    }
    
    # Print summary
    if verbose:
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Episodes: {n_episodes}")
        print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"Mean Episode Length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f} steps")
        print(f"Mean Biomass Gain: {stats['mean_biomass']:.2f} ± {stats['std_biomass']:.2f} g")
        print(f"Mean Stresses: Water={stats['mean_stress_water']:.3f}, "
              f"Temp={stats['mean_stress_temp']:.3f}, "
              f"Nutrient={stats['mean_stress_nutrient']:.3f}")
        print(f"% Reaching Max Steps: {stats['pct_reaching_max_steps']:.1f}%")
        print(f"\nTermination Reasons:")
        for reason, count in stats['termination_reasons'].items():
            print(f"  {reason}: {count} ({count/n_episodes*100:.1f}%)")
        print(f"\n{'='*80}")
        print("NEW METRICS:")
        print(f"% Reaching 336 Steps: {stats['pct_reaching_336_steps']:.1f}%")
        print(f"Avg Daytime Photosynthesis (hours 6-18): {stats['avg_daytime_photosynthesis']:.6f}")
        print(f"Avg Water Action (High VPD windows): {stats['avg_water_action_high_vpd']:.4f}")
        print(f"Avg Water Action (High Temp >28°C): {stats['avg_water_action_high_temp']:.4f}")
        print(f"{'='*80}\n")
        
        # Print detailed trajectory sample
        if detailed_trajectory:
            print("Sample Trajectory (first 5 steps):")
            for step_data in detailed_trajectory[:5]:
                print(f"\nStep {step_data['step']}:")
                print(f"  Reward: {step_data['reward']:.3f}")
                if 'true_state' in step_data:
                    plant = step_data['true_state'].get('plant', {})
                    env_state = step_data['true_state'].get('env', {})
                    print(f"  True State - Plant biomass: {plant.get('total_biomass', 0):.3f}g, "
                          f"Moisture: {plant.get('moisture', 0):.3f}, "
                          f"T: {env_state.get('T_middle', 0):.1f}°C")
                if 'applied_action' in step_data:
                    action = step_data['applied_action']
                    print(f"  Applied Action - Water: {action.get('water_total', 0):.3f}, "
                          f"Fan: {action.get('fan', False)}, "
                          f"Heater: {action.get('heater', 0):.3f}")
            print()
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained PPO policy')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained PPO model (.zip file)')
    parser.add_argument('--n_episodes', type=int, default=20,
                       help='Number of episodes to run (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic policy (default: True)')
    parser.add_argument('--stochastic', action='store_true',
                       help='Use stochastic policy (overrides --deterministic)')
    
    args = parser.parse_args()
    
    deterministic = not args.stochastic
    
    stats = evaluate_policy(
        model_path=args.model,
        n_episodes=args.n_episodes,
        seed=args.seed,
        deterministic=deterministic,
        verbose=True
    )
    
    return stats


if __name__ == '__main__':
    main()

