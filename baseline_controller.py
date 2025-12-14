#!/usr/bin/env python3
"""
Baseline Controller for Digital Twin Validation
A simple rule-based controller to verify environment works correctly.

This controller should be able to keep plants alive for 10+ days if environment is working.
If it can't, there's a bug in the environment logic.
"""

import numpy as np
import argparse
from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from rl.gym_env import EnhancedDigitalTwinEnv as DigitalTwinEnv
except ImportError:
    print("Could not import gym_env. Make sure you're in the correct directory.")
    sys.exit(1)


class BaselineController:
    """
    Simple rule-based controller using plant physiology knowledge.
    
    Rules:
    1. Water when moisture < 0.4 (before stress threshold)
    2. Turn on fan when temp > 28°C
    3. Close shield when light is high (>0.8) and temp > 26°C
    4. Turn on heater when temp < 18°C
    """
    
    def __init__(self, aggressive_watering=False):
        self.aggressive_watering = aggressive_watering
        self.name = "Baseline (Aggressive)" if aggressive_watering else "Baseline (Conservative)"
    
    def predict(self, obs, deterministic=True):
        """
        Predict action based on observations.
        
        Observation structure (11 features):
        [0] biomass_norm
        [1] moisture
        [2] nutrient
        [3] LAI_norm
        [4] temp_norm (0-1, represents 0-40°C)
        [5] light_norm
        [6] stress_water
        [7] stress_temp
        [8] stress_nutrient
        [9] hour_sin
        [10] hour_cos
        """
        # Extract relevant features
        moisture = obs[1]
        temp_norm = obs[4]
        light_norm = obs[5]
        stress_water = obs[6]
        stress_temp = obs[7]
        
        # Denormalize temperature (assuming 0-1 maps to 0-40°C)
        temp = temp_norm * 40.0
        
        # === WATERING LOGIC ===
        if self.aggressive_watering:
            # Water when moisture < 0.45 or water stress detected
            water = 0.6 if (moisture < 0.45 or stress_water < 0.8) else 0.0
        else:
            # Conservative: water only when moisture < 0.35
            water = 0.4 if (moisture < 0.35) else 0.0
        
        # === FAN LOGIC ===
        # Turn on fan when too hot OR to reduce humidity
        fan = 1.0 if (temp > 28.0 or stress_temp < 0.7) else 0.0
        
        # === SHIELD LOGIC ===
        # Close shield partially when hot and bright (prevent overheating)
        if temp > 26.0 and light_norm > 0.7:
            shield_delta = 0.3  # Move toward closed
        elif temp < 22.0 and light_norm > 0.5:
            shield_delta = -0.3  # Move toward open (need warmth)
        else:
            shield_delta = 0.0  # No change
        
        # === HEATER LOGIC ===
        # Turn on heater when too cold
        if temp < 18.0:
            heater = 0.8
        elif temp < 20.0:
            heater = 0.4
        else:
            heater = 0.0
        
        action = np.array([water, fan, shield_delta, heater], dtype=np.float32)
        return action, None


def run_baseline_evaluation(n_episodes=10, aggressive=False):
    """Run baseline controller and collect statistics"""
    print("="*70)
    print("BASELINE CONTROLLER EVALUATION")
    print("="*70)
    
    # Create environment
    env = DigitalTwinEnv()
    controller = BaselineController(aggressive_watering=aggressive)
    
    print(f"\nController: {controller.name}")
    print(f"Episodes: {n_episodes}")
    print(f"Max steps per episode: {env.max_steps}")
    print("\n" + "-"*70)
    
    results = {
        'rewards': [],
        'lengths': [],
        'final_biomass': [],
        'deaths': [],
        'death_reasons': [],
        'water_used': [],
        'growth_achieved': []
    }
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        step = 0
        
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        print("-" * 50)
        
        while not done:
            # Get action from baseline controller
            action, _ = controller.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # Print progress every 24 hours (24 steps)
            if step % 24 == 0:
                plant_state = info['plant']
                env_state = info['env']
                print(f"  Day {step // 24}: "
                      f"Biomass={plant_state['biomass_total']:.2f}g  "
                      f"Moisture={plant_state['soil_moisture']:.2f}  "
                      f"Temp={env_state['T']:.1f}°C  "
                      f"Reward={episode_reward / step:.2f}")
            
            done = terminated or truncated
        
        # Collect episode statistics
        plant_final = info['plant']
        results['rewards'].append(episode_reward)
        results['lengths'].append(step)
        results['final_biomass'].append(plant_final['biomass_total'])
        results['deaths'].append(1 if info.get('death_reason') else 0)
        results['death_reasons'].append(info.get('death_reason', 'completed'))
        results['water_used'].append(info.get('cumulative_water_used', 0))
        results['growth_achieved'].append(info.get('cumulative_growth', 0))
        
        print(f"  FINAL: Steps={step}  Reward={episode_reward:.1f}  "
              f"Biomass={plant_final['biomass_total']:.2f}g  "
              f"Status={info.get('death_reason', 'survived')}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Mean Episode Reward:    {np.mean(results['rewards']):.2f} ± {np.std(results['rewards']):.2f}")
    print(f"Mean Survival Time:     {np.mean(results['lengths']):.1f} / {env.max_steps} steps")
    print(f"Mean Final Biomass:     {np.mean(results['final_biomass']):.2f}g")
    print(f"Mean Growth:            {np.mean(results['growth_achieved']):.2f}g")
    print(f"Death Rate:             {np.mean(results['deaths'])*100:.1f}%")
    print(f"Mean Water Used:        {np.mean(results['water_used']):.3f}L")
    print(f"Water Use Efficiency:   {np.mean(results['growth_achieved'])/max(0.01, np.mean(results['water_used'])):.2f} g/L")
    
    print("\nDeath Reasons:")
    death_counts = {}
    for reason in results['death_reasons']:
        death_counts[reason] = death_counts.get(reason, 0) + 1
    for reason, count in sorted(death_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count} ({count/n_episodes*100:.1f}%)")
    
    print("\n" + "="*70)
    
    # Validation checks
    print("\nVALIDATION CHECKS:")
    print("-" * 70)
    
    survival_rate = 1 - np.mean(results['deaths'])
    mean_survival = np.mean(results['lengths'])
    mean_growth = np.mean(results['growth_achieved'])
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: Survival rate
    checks_total += 1
    if survival_rate >= 0.5:
        print("✅ PASS: Survival rate >= 50%")
        checks_passed += 1
    else:
        print("❌ FAIL: Survival rate < 50% (environment may have bugs)")
    
    # Check 2: Mean survival time
    checks_total += 1
    if mean_survival >= 168:  # 7 days
        print("✅ PASS: Mean survival >= 7 days")
        checks_passed += 1
    else:
        print("❌ FAIL: Mean survival < 7 days (check stress/death conditions)")
    
    # Check 3: Growth achieved
    checks_total += 1
    if mean_growth >= 2.0:
        print("✅ PASS: Mean growth >= 2.0g")
        checks_passed += 1
    else:
        print("❌ FAIL: Mean growth < 2.0g (check photosynthesis/growth logic)")
    
    # Check 4: Positive rewards
    checks_total += 1
    if np.mean(results['rewards']) > -100:
        print("✅ PASS: Mean reward > -100 (reward function reasonable)")
        checks_passed += 1
    else:
        print("❌ FAIL: Mean reward < -100 (reward function may be too harsh)")
    
    print("-" * 70)
    print(f"OVERALL: {checks_passed}/{checks_total} checks passed")
    
    if checks_passed == checks_total:
        print("🎉 Environment appears to be working correctly!")
        print("   You can proceed with RL training.")
    elif checks_passed >= checks_total // 2:
        print("⚠️  Environment partially working but needs tuning.")
        print("   Review failed checks before full training.")
    else:
        print("🚨 Environment has critical issues!")
        print("   Fix bugs before attempting RL training.")
    
    print("="*70 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run baseline controller evaluation')
    parser.add_argument('--episodes', type=int, default=10, 
                       help='Number of episodes to run')
    parser.add_argument('--aggressive', action='store_true',
                       help='Use aggressive watering strategy')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with 3 episodes')
    
    args = parser.parse_args()
    
    n_episodes = 3 if args.quick else args.episodes
    
    try:
        results = run_baseline_evaluation(n_episodes=n_episodes, aggressive=args.aggressive)
        
        # Optionally save results
        import json
        output_path = Path('./baseline_results.json')
        with open(output_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {
                k: [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in vals]
                for k, vals in results.items()
            }
            json.dump(json_results, f, indent=2)
        print(f"Results saved to {output_path}")
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\n\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())