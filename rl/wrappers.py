# rl/wrappers.py
"""
PPO-ready wrappers for DigitalTwinEnv.
Refactored for numerical stability.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class NormalizeObservation(gym.ObservationWrapper):
    """
    Normalize observations using running statistics.
    Epsilon increased to 1e-6 to prevent div/0 errors on constant features.
    """
    
    def __init__(self, env, epsilon=1e-6):
        super().__init__(env)
        self.epsilon = epsilon
        self.running_mean = np.zeros(env.observation_space.shape, dtype=np.float32)
        self.running_var = np.ones(env.observation_space.shape, dtype=np.float32)
        self.count = 0
        
    def observation(self, obs):
        self.count += 1
        delta = obs - self.running_mean
        self.running_mean += delta / self.count
        delta2 = obs - self.running_mean
        self.running_var += delta * delta2
        
        std = np.sqrt(self.running_var / self.count + self.epsilon)
        normalized = (obs - self.running_mean) / std
        
        # Clip to prevent extreme values
        normalized = np.clip(normalized, -10.0, 10.0)
        return normalized.astype(np.float32)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


class ClipAction(gym.ActionWrapper):
    """Clip actions to valid range."""
    def __init__(self, env):
        super().__init__(env)
    
    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)


class ScaleReward(gym.RewardWrapper):
    """Scale rewards to reasonable range for PPO."""
    def __init__(self, env, scale=0.1):
        super().__init__(env)
        self.scale = scale
    
    def reward(self, reward):
        return float(reward * self.scale)


class EpisodeStats(gym.Wrapper):
    """Track episode statistics."""
    def __init__(self, env):
        super().__init__(env)
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_stats = {}
    
    def reset(self, **kwargs):
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_stats = {}
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        self.episode_length += 1
        
        if 'plant_dead' in info:
            self.episode_stats['plant_dead'] = info['plant_dead']
        if 'death_reason' in info:
            self.episode_stats['death_reason'] = info['death_reason']
        
        if terminated or truncated:
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.episode_length,
                **self.episode_stats
            }
        
        return obs, reward, terminated, truncated, info


class FrameStack(gym.ObservationWrapper):
    """Stack consecutive frames."""
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
        
        low = np.tile(env.observation_space.low, n_frames)
        high = np.tile(env.observation_space.high, n_frames)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self.observation(obs), info
    
    def observation(self, obs):
        self.frames.append(obs)
        return np.concatenate(list(self.frames), axis=0).astype(np.float32)


def make_env(cfg=None, use_wrappers=True, use_framestack=False, n_frames=4):
    """Factory function to create and wrap environment."""
    from rl.gym_env import ImprovedDigitalTwinEnv as DigitalTwinEnv
    
    env = DigitalTwinEnv(cfg=cfg)
    
    if use_wrappers:
        env = ClipAction(env)
        env = EpisodeStats(env)
        env = NormalizeObservation(env)
        env = ScaleReward(env, scale=0.1)
        
        if use_framestack:
            env = FrameStack(env, n_frames=n_frames)
    
    return env