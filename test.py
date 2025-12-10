from stable_baselines3 import PPO
from rl.gym_env import ImprovedDigitalTwinEnv
import numpy as np

env = ImprovedDigitalTwinEnv()
model = PPO.load('ppo_best/best_model')  # Load best checkpoint

for ep in range(3):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
    
    print(f'Episode {ep+1}: Reward={total_reward:.1f}, Steps={steps}, Growth={info["cumulative_growth"]:.2f}g')