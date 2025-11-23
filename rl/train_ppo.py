# rl/train_ppo.py
"""
Train PPO agent on the DigitalTwinEnv.

Usage:
    python rl/train_ppo.py --demo      # quick 2000-step demo
    python rl/train_ppo.py             # full training

This script creates the environment, trains PPO, and saves the model.
"""

import argparse
from stable_baselines3 import PPO
from rl.gym_env import DigitalTwinEnv

def main(args):
    env = DigitalTwinEnv()

    if args.demo:
        print("Running quick PPO demo training...")
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=2000)
        model.save('ppo_demo')
        print("Demo model saved as ppo_demo")
    else:
        print("Running full PPO training (200k steps)...")
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=200000)
        model.save('ppo_full')
        print("Full model saved as ppo_full")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()
    main(args)
