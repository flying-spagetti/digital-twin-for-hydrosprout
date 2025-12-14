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
    Normalize observations using Welford's online algorithm for running statistics.
    
    Uses correct Welford algorithm:
    - Maintains mean, M2 (sum of squared differences), and count
    - Variance = M2 / max(count-1, 1)
    - Std = sqrt(variance + epsilon)
    
    Epsilon prevents div/0 errors on constant features.
    Automatically handles observation space dimension changes.
    """
    
    def __init__(self, env, epsilon=1e-6):
        super().__init__(env)
        self.epsilon = epsilon
        self.running_mean = np.zeros(env.observation_space.shape, dtype=np.float32)
        self.M2 = np.zeros(env.observation_space.shape, dtype=np.float32)  # Sum of squared differences
        self.count = 0
        self.expected_shape = env.observation_space.shape
        
    def _check_and_reset_stats(self, obs_shape):
        """Reset running statistics if observation shape changed."""
        if obs_shape != self.expected_shape:
            # Observation space changed, reinitialize stats
            self.running_mean = np.zeros(obs_shape, dtype=np.float32)
            self.M2 = np.zeros(obs_shape, dtype=np.float32)
            self.count = 0
            self.expected_shape = obs_shape
        
    def observation(self, obs):
        # Check if observation shape matches expected shape
        obs_shape = obs.shape if isinstance(obs, np.ndarray) else np.array(obs).shape
        self._check_and_reset_stats(obs_shape)
        
        # Welford's online algorithm
        self.count += 1
        delta = obs - self.running_mean
        self.running_mean += delta / self.count
        delta2 = obs - self.running_mean
        self.M2 += delta * delta2
        
        # Calculate variance and std using Welford formula
        # Variance = M2 / (count - 1) for sample variance, or M2 / count for population
        # Use (count - 1) to match standard sample variance formula
        var = self.M2 / max(self.count - 1, 1)
        std = np.sqrt(var + self.epsilon)
        
        # Normalize
        normalized = (obs - self.running_mean) / std
        
        # Clip to prevent extreme values
        normalized = np.clip(normalized, -10.0, 10.0)
        return normalized.astype(np.float32)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


class ClipAction(gym.ActionWrapper):
    """Clip actions to valid range. Handles both Box and Dict action spaces."""
    def __init__(self, env):
        super().__init__(env)
    
    def action(self, action):
        if isinstance(self.action_space, spaces.Dict):
            # For Dict spaces, clip each component individually
            clipped = {}
            for key, space in self.action_space.spaces.items():
                if isinstance(space, spaces.Box):
                    clipped[key] = np.clip(action[key], space.low, space.high)
                else:
                    # For Discrete/MultiBinary, just pass through
                    clipped[key] = action[key]
            return clipped
        else:
            # For Box spaces, use standard clipping
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


class FlattenDictAction(gym.ActionWrapper):
    """
    Flatten Dict action space to Box for PPO compatibility.
    
    Converts Dict action space to a single Box space by concatenating all components.
    Order: water_total, fan, shield_delta, heater, peltier_controls, dose_N, dose_P, dose_K, pH_adjust, nozzle_mask, co2_inject
    """
    def __init__(self, env):
        super().__init__(env)
        
        if not isinstance(env.action_space, spaces.Dict):
            # Already not a Dict, no need to wrap
            return
        
        # Build flattened action space
        lows = []
        highs = []
        
        # Define order of actions
        self.action_order = [
            'water_total', 'fan', 'shield_delta', 'heater',
            'peltier_controls', 'dose_N', 'dose_P', 'dose_K', 
            'pH_adjust', 'nozzle_mask', 'co2_inject'
        ]
        
        for key in self.action_order:
            if key in env.action_space.spaces:
                space = env.action_space.spaces[key]
                if isinstance(space, spaces.Box):
                    lows.extend(space.low.flatten())
                    highs.extend(space.high.flatten())
                elif isinstance(space, spaces.Discrete):
                    lows.append(0)
                    highs.append(space.n - 1)
                elif isinstance(space, spaces.MultiBinary):
                    lows.extend([0] * space.n)
                    highs.extend([1] * space.n)
        
        self.action_space = spaces.Box(
            low=np.array(lows, dtype=np.float32),
            high=np.array(highs, dtype=np.float32),
            dtype=np.float32
        )
        
        # Store original action space structure for conversion
        self.original_action_space = env.action_space
    
    def action(self, action):
        """Convert flattened Box action back to Dict action."""
        if not isinstance(self.env.action_space, spaces.Dict):
            return action
        
        # FIXED: Ensure action is numpy array for slicing/reshaping
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        
        # Convert flattened array back to Dict
        dict_action = {}
        idx = 0
        
        for key in self.action_order:
            if key not in self.original_action_space.spaces:
                continue
                
            space = self.original_action_space.spaces[key]
            
            if isinstance(space, spaces.Box):
                size = np.prod(space.shape)
                values = action[idx:idx+size].reshape(space.shape)
                dict_action[key] = values
                idx += size
            elif isinstance(space, spaces.Discrete):
                dict_action[key] = int(np.clip(action[idx], 0, space.n - 1))
                idx += 1
            elif isinstance(space, spaces.MultiBinary):
                size = space.n
                values = (action[idx:idx+size] > 0.5).astype(np.int32)
                dict_action[key] = values
                idx += size
        
        return dict_action


def make_env(cfg=None, use_wrappers=True, use_framestack=False, n_frames=4):
    """Factory function to create and wrap environment."""
    from rl.gym_env import EnhancedDigitalTwinEnv as DigitalTwinEnv
    
    env = DigitalTwinEnv(cfg=cfg)
    
    # Flatten Dict action space first (before other wrappers)
    if isinstance(env.action_space, spaces.Dict):
        env = FlattenDictAction(env)
    
    if use_wrappers:
        env = ClipAction(env)
        env = EpisodeStats(env)
        env = NormalizeObservation(env)
        env = ScaleReward(env, scale=0.1)
        
        if use_framestack:
            env = FrameStack(env, n_frames=n_frames)
    
    return env