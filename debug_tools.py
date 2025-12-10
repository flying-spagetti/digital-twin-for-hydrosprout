#!/usr/bin/env python3
"""
Debug Tools for Digital Twin System
Comprehensive validation and diagnostic utilities
"""

import numpy as np
import argparse
from pathlib import Path
import sys
import json

# Add project root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def validate_stress_factors():
    """Test #1: Verify stress factors are being updated"""
    print("\n" + "="*70)
    print("TEST 1: Stress Factor Updates")
    print("="*70)
    
    try:
        from sim.plant_SFPM import HydroponicPlantFSPM
        plant = HydroponicPlantFSPM(initial_biomass=1.0)
        
        print("\n1. Testing DRY conditions (moisture=0.2)")
        plant.soil_moisture = 0.2  # Dry
        result = plant.step(light=0.5, temp=22.0)
        
        print(f"   Moisture: {result['soil_moisture']:.2f}")
        print(f"   Water stress: {result['stress_water']:.2f}")
        
        if result['stress_water'] < 0.5:
            print("   ✅ PASS: Water stress correctly detected")
            test1_pass = True
        else:
            print("   ❌ FAIL: Water stress should be < 0.5, got {:.2f}".format(result['stress_water']))
            print("   → BUG: Stress factors not being updated!")
            test1_pass = False
        
        print("\n2. Testing HOT conditions (temp=34°C)")
        plant.reset()
        result = plant.step(light=0.5, temp=34.0)  # Near max temp
        
        print(f"   Temperature: {result['stress_temp']:.2f}")
        
        if result['stress_temp'] < 0.5:
            print("   ✅ PASS: Temperature stress correctly detected")
            test2_pass = True
        else:
            print("   ❌ FAIL: Temp stress should be < 0.5 at 34°C")
            test2_pass = False
        
        print("\n3. Testing OPTIMAL conditions (moisture=0.5, temp=25°C)")
        plant.reset()
        result = plant.step(light=0.7, temp=25.0)
        
        print(f"   Moisture: {result['soil_moisture']:.2f}")
        print(f"   Temperature: 25.0°C")
        print(f"   Water stress: {result['stress_water']:.2f}")
        print(f"   Temp stress: {result['stress_temp']:.2f}")
        
        if result['stress_water'] > 0.9 and result['stress_temp'] > 0.9:
            print("   ✅ PASS: Optimal conditions show no stress")
            test3_pass = True
        else:
            print("   ❌ FAIL: Optimal conditions should show high stress values")
            test3_pass = False
        
        overall = test1_pass and test2_pass and test3_pass
        print("\n" + "-"*70)
        print(f"STRESS FACTORS: {'✅ PASS' if overall else '❌ FAIL'}")
        return overall
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_light_photosynthesis():
    """Test #2: Verify light → photosynthesis → growth chain"""
    print("\n" + "="*70)
    print("TEST 2: Light → Photosynthesis → Growth Chain")
    print("="*70)
    
    try:
        from sim.plant_SFPM import HydroponicPlantFSPM
        plant = HydroponicPlantFSPM(initial_biomass=1.0)
        
        # Test with normalized light input
        print("\n1. Testing normalized light (0.8) → PAR conversion")
        result1 = plant.step(light=0.8, temp=25.0)
        print(f"   Input light: 0.8 (normalized)")
        print(f"   Photosynthesis rate: {result1['photosynthesis_rate']:.3f} g CO2/m²/h")
        print(f"   Net growth: {result1['net_growth']:.4f} g")
        
        if result1['photosynthesis_rate'] > 0.1:
            print("   ✅ PASS: Photosynthesis occurring")
            test1_pass = True
        else:
            print("   ❌ FAIL: No photosynthesis detected")
            print("   → Check light unit conversion (should use PAR = light * 1500)")
            test1_pass = False
        
        # Test growth accumulation over 24 hours
        print("\n2. Testing 24-hour growth accumulation")
        plant.reset()
        initial_biomass = plant.organs.B_leaf.sum() + plant.organs.B_stem.sum() + plant.organs.B_root.sum()
        
        total_photo = 0.0
        total_growth = 0.0
        for hour in range(24):
            # Simulate day/night cycle
            if 6 <= hour <= 18:  # Day
                light = 0.6 + 0.3 * np.sin((hour - 6) / 12.0 * np.pi)
            else:  # Night
                light = 0.0
            
            result = plant.step(light=light, temp=23.0)
            total_photo += result['photosynthesis_rate']
            total_growth += result['net_growth']
        
        final_biomass = plant.organs.B_leaf.sum() + plant.organs.B_stem.sum() + plant.organs.B_root.sum()
        biomass_gain = final_biomass - initial_biomass
        
        print(f"   Initial biomass: {initial_biomass:.3f}g")
        print(f"   Final biomass: {final_biomass:.3f}g")
        print(f"   Biomass gain: {biomass_gain:.3f}g")
        print(f"   Total photosynthesis: {total_photo:.2f} g CO2")
        print(f"   Total net growth: {total_growth:.3f}g")
        
        if biomass_gain > 0.01:  # At least 10mg growth per day
            print("   ✅ PASS: Biomass accumulating over time")
            test2_pass = True
        else:
            print("   ❌ FAIL: No biomass gain over 24 hours")
            test2_pass = False
        
        overall = test1_pass and test2_pass
        print("\n" + "-"*70)
        print(f"PHOTOSYNTHESIS: {'✅ PASS' if overall else '❌ FAIL'}")
        return overall
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_water_action_effect():
    """Test #3: Verify water action → moisture increase"""
    print("\n" + "="*70)
    print("TEST 3: Water Action → Moisture Effect")
    print("="*70)
    
    try:
        from rl.gym_env import ImprovedDigitalTwinEnv as DigitalTwinEnv
        env = DigitalTwinEnv()
        
        print("\n1. Testing NO water action")
        obs, _ = env.reset()
        initial_moisture = obs[1]  # Index 1 is moisture
        
        # Take 5 steps with no water
        for _ in range(5):
            action = np.array([0.0, 0.0, 0.0, 0.0])  # No water
            obs, _, _, _, _ = env.step(action)
        
        final_moisture = obs[1]
        print(f"   Initial moisture: {initial_moisture:.3f}")
        print(f"   Final moisture: {final_moisture:.3f}")
        print(f"   Change: {final_moisture - initial_moisture:.3f}")
        
        if final_moisture < initial_moisture:
            print("   ✅ PASS: Moisture decreases without watering")
            test1_pass = True
        else:
            print("   ❌ FAIL: Moisture should decrease")
            test1_pass = False
        
        print("\n2. Testing WITH water action")
        obs, _ = env.reset()
        initial_moisture = obs[1]
        
        # Take 5 steps WITH water
        for _ in range(5):
            action = np.array([0.8, 0.0, 0.0, 0.0])  # Max water
            obs, _, _, _, _ = env.step(action)
        
        final_moisture = obs[1]
        print(f"   Initial moisture: {initial_moisture:.3f}")
        print(f"   Final moisture: {final_moisture:.3f}")
        print(f"   Change: {final_moisture - initial_moisture:.3f}")
        
        if final_moisture > initial_moisture:
            print("   ✅ PASS: Moisture increases with watering")
            test2_pass = True
        else:
            print("   ❌ FAIL: Moisture should increase with watering")
            print("   → Check hardware water delivery and plant water balance")
            test2_pass = False
        
        overall = test1_pass and test2_pass
        print("\n" + "-"*70)
        print(f"WATER ACTION: {'✅ PASS' if overall else '❌ FAIL'}")
        return overall
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_reward_structure():
    """Test #4: Verify reward function gives sensible values"""
    print("\n" + "="*70)
    print("TEST 4: Reward Function Structure")
    print("="*70)
    
    try:
        from rl.gym_env import ImprovedDigitalTwinEnv as DigitalTwinEnv
        env = DigitalTwinEnv()
        
        print("\n1. Testing normal operation")
        obs, _ = env.reset()
        rewards = []
        
        for _ in range(24):
            # Reasonable actions
            action = np.array([0.2, 0.0, 0.0, 0.0])
            obs, reward, _, _, _ = env.step(action)
            rewards.append(reward)
        
        mean_reward = np.mean(rewards)
        print(f"   Mean reward over 24 steps: {mean_reward:.2f}")
        print(f"   Reward range: [{min(rewards):.2f}, {max(rewards):.2f}]")
        
        if -20 < mean_reward < 20:
            print("   ✅ PASS: Rewards in reasonable range")
            test1_pass = True
        else:
            print("   ❌ FAIL: Rewards too extreme")
            print("   → Check reward scaling and components")
            test1_pass = False
        
        print("\n2. Testing growth reward")
        obs, _ = env.reset()
        prev_biomass = obs[0]
        
        # Give lots of light and water for growth
        for _ in range(48):  # 2 days
            action = np.array([0.3, 0.0, 0.0, 0.0])
            obs, reward, _, _, _ = env.step(action)
        
        current_biomass = obs[0]
        print(f"   Biomass change: {current_biomass - prev_biomass:.4f}")
        
        if current_biomass > prev_biomass:
            print("   ✅ PASS: Biomass increasing")
            test2_pass = True
        else:
            print("   ❌ FAIL: Biomass should increase with good conditions")
            test2_pass = False
        
        overall = test1_pass and test2_pass
        print("\n" + "-"*70)
        print(f"REWARD FUNCTION: {'✅ PASS' if overall else '❌ FAIL'}")
        return overall
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_curriculum():
    """Test #5: Verify curriculum config is applied"""
    print("\n" + "="*70)
    print("TEST 5: Curriculum Configuration")
    print("="*70)
    
    try:
        from rl.gym_env import ImprovedDigitalTwinEnv as DigitalTwinEnv
        from rl.curriculum import CurriculumScheduler, CurriculumWrapper
        
        print("\n1. Creating curriculum scheduler")
        curriculum = CurriculumScheduler(total_timesteps=100000)
        print(f"   Initial stage: {curriculum.current_stage}")
        print(f"   Total stages: {len(curriculum.stages)}")
        
        print("\n2. Testing curriculum application")
        base_env = DigitalTwinEnv()
        env = CurriculumWrapper(base_env, curriculum)
        
        # Reset with warmup stage
        curriculum.update(0)
        obs, _ = env.reset()
        print(f"   Stage at timestep 0: {curriculum.current_stage}")
        
        # Advance to easy stage
        curriculum.update(15000)
        obs, _ = env.reset()
        print(f"   Stage at timestep 15000: {curriculum.current_stage}")
        
        if curriculum.current_stage == 'easy':
            print("   ✅ PASS: Curriculum stages advancing")
            test1_pass = True
        else:
            print("   ❌ FAIL: Should be in 'easy' stage at 15000 steps")
            test1_pass = False
        
        print("\n" + "-"*70)
        print(f"CURRICULUM: {'✅ PASS' if test1_pass else '❌ FAIL'}")
        return test1_pass
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_validations():
    """Run all validation tests"""
    print("\n" + "="*70)
    print("DIGITAL TWIN SYSTEM VALIDATION")
    print("Running comprehensive diagnostic tests...")
    print("="*70)
    
    results = {}
    
    # Run each test
    results['stress_factors'] = validate_stress_factors()
    results['photosynthesis'] = validate_light_photosynthesis()
    results['water_action'] = validate_water_action_effect()
    results['reward'] = validate_reward_structure()
    results['curriculum'] = validate_curriculum()
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.upper():20s}: {status}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print("-"*70)
    print(f"OVERALL: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! System is ready for training.")
        return 0
    elif passed_count >= 3:
        print("\n⚠️  Most tests passed, but some issues remain.")
        print("Review failed tests before full training run.")
        return 1
    else:
        print("\n🚨 Critical issues detected! Fix bugs before training.")
        return 2


def analyze_training_logs(log_dir):
    """Analyze training logs and provide diagnostics"""
    print("\n" + "="*70)
    print(f"ANALYZING TRAINING LOGS: {log_dir}")
    print("="*70)
    
    log_path = Path(log_dir)
    
    # Check for summary file
    summary_file = log_path / 'summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print("\nTraining Summary:")
        print("-"*70)
        print(f"Total Episodes: {summary.get('total_episodes', 'N/A')}")
        print(f"Mean Reward (last 100): {summary.get('mean_reward', 0):.2f}")
        print(f"Mean Growth (last 100): {summary.get('mean_growth', 0):.2f}g")
        print(f"Mean Survival (last 100): {summary.get('mean_survival', 0):.1f} steps")
        print(f"Death Rate (last 100): {summary.get('death_rate', 0)*100:.1f}%")
        print(f"Water Efficiency: {summary.get('mean_water_efficiency', 0):.2f} g/L")
        
        # Diagnostics
        print("\nDiagnostics:")
        print("-"*70)
        
        if summary.get('mean_reward', -999) < -100:
            print("❌ Very negative rewards - check reward function scaling")
        elif summary.get('mean_reward', -999) > 50:
            print("✅ Good rewards - agent learning well")
        
        if summary.get('death_rate', 1.0) > 0.5:
            print("❌ High death rate - agent struggling to keep plant alive")
        elif summary.get('death_rate', 1.0) < 0.2:
            print("✅ Low death rate - agent learning to maintain plant")
        
        if summary.get('mean_growth', 0) < 1.0:
            print("❌ Low growth - check photosynthesis and growth logic")
        elif summary.get('mean_growth', 0) > 5.0:
            print("✅ Good growth - agent optimizing effectively")
    
    else:
        print("No summary.json found. Make sure training has completed.")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Digital Twin Debug Tools')
    parser.add_argument('--validate', action='store_true',
                       help='Run all validation tests')
    parser.add_argument('--test-stress', action='store_true',
                       help='Test stress factor updates only')
    parser.add_argument('--test-photo', action='store_true',
                       help='Test photosynthesis only')
    parser.add_argument('--test-water', action='store_true',
                       help='Test water action effects only')
    parser.add_argument('--test-reward', action='store_true',
                       help='Test reward function only')
    parser.add_argument('--test-curriculum', action='store_true',
                       help='Test curriculum only')
    parser.add_argument('--analyze', type=str, metavar='LOG_DIR',
                       help='Analyze training logs from directory')
    
    args = parser.parse_args()
    
    if args.validate:
        return run_all_validations()
    elif args.test_stress:
        return 0 if validate_stress_factors() else 1
    elif args.test_photo:
        return 0 if validate_light_photosynthesis() else 1
    elif args.test_water:
        return 0 if validate_water_action_effect() else 1
    elif args.test_reward:
        return 0 if validate_reward_structure() else 1
    elif args.test_curriculum:
        return 0 if validate_curriculum() else 1
    elif args.analyze:
        analyze_training_logs(args.analyze)
        return 0
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    exit(main())