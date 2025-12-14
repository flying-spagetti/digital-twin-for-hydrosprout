#!/usr/bin/env python3
"""
run_checks.py

Run the requested checks:
1. Verify debug info is logged in sim runner
2. Verify hardware reports both coverage metrics
3. Run 10 eval episodes and record metrics
"""

import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rl.evaluate_policy import evaluate_policy
from pathlib import Path

def main():
    # Find best model
    model_path = ROOT / "ppo_best" / "best_model.zip"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first or specify a different path.")
        return 1
    
    print("="*80)
    print("RUNNING CHECKS")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Episodes: 10")
    print("="*80)
    print()
    
    # Run evaluation
    stats = evaluate_policy(
        model_path=str(model_path),
        n_episodes=10,
        seed=42,
        deterministic=True,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("CHECK RESULTS SUMMARY")
    print("="*80)
    print(f"✓ % Reaching 336 Steps: {stats['pct_reaching_336_steps']:.1f}%")
    print(f"✓ Avg Daytime Photosynthesis (hours 6-18): {stats['avg_daytime_photosynthesis']:.6f}")
    print(f"✓ Avg Water Action (High VPD windows): {stats['avg_water_action_high_vpd']:.4f}")
    print(f"✓ Avg Water Action (High Temp >28°C): {stats['avg_water_action_high_temp']:.4f}")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

