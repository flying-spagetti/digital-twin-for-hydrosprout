# rl/curriculum.py
import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional

class CurriculumScheduler:
    """
    Manages curriculum learning progression.
    
    Stages:
    1. Warmup: Perfect conditions. Agent learns basic mechanics (Watering = Moisture).
    2. Easy: Minor noise. Agent learns to correct small deviations.
    3. Medium: Variable conditions. Introduction of heat stress.
    4. Hard: Challenging conditions. High noise, requiring robust control.
    5. Expert: Realistic conditions, disturbances, full complexity.
    """
    
    def __init__(self, total_timesteps: int, stages: Optional[Dict] = None):
        self.total_timesteps = total_timesteps
        self.current_timesteps = 0
        
        if stages is None:
            self.stages = {
                'warmup': {
                    'timestep_range': (0, int(0.10 * total_timesteps)),
                    'temp_noise_std': 0.1,
                    'moisture_noise_std': 0.01,
                    'initial_temp': 22.0,
                    'initial_moisture': 0.50,
                    'disturbance_prob': 0.0,
                    'episode_length_days': 7,
                },
                'easy': {
                    'timestep_range': (int(0.10 * total_timesteps), int(0.30 * total_timesteps)),
                    'temp_noise_std': 0.5,
                    'moisture_noise_std': 0.05,
                    'initial_temp': 22.0,
                    'initial_moisture': 0.45,
                    'disturbance_prob': 0.0,
                    'episode_length_days': 10,
                },
                'medium': {
                    'timestep_range': (int(0.30 * total_timesteps), int(0.60 * total_timesteps)),
                    'temp_noise_std': 1.5,
                    'moisture_noise_std': 0.10,
                    'initial_temp': 24.0,
                    'initial_moisture': 0.40,
                    'disturbance_prob': 0.05,
                    'episode_length_days': 14,
                },
                'hard': {
                    'timestep_range': (int(0.60 * total_timesteps), int(0.85 * total_timesteps)),
                    'temp_noise_std': 2.5,
                    'moisture_noise_std': 0.15,
                    'initial_temp': 26.0,
                    'initial_moisture': 0.35,
                    'disturbance_prob': 0.15,
                    'episode_length_days': 14,
                },
                'expert': {
                    'timestep_range': (int(0.85 * total_timesteps), total_timesteps),
                    'temp_noise_std': 4.0,
                    'moisture_noise_std': 0.20,
                    'initial_temp': 28.0,
                    'initial_moisture': 0.30,
                    'disturbance_prob': 0.3,
                    'episode_length_days': 14,
                },
            }
        else:
            self.stages = stages
        
        self.current_stage = 'warmup'
    
    def update(self, timesteps: int):
        self.current_timesteps = timesteps
        for stage_name, stage_config in self.stages.items():
            start, end = stage_config['timestep_range']
            if start <= timesteps < end:
                self.current_stage = stage_name
                break
        else:
            self.current_stage = 'expert'
    
    def get_config(self) -> Dict[str, Any]:
        return self.stages[self.current_stage].copy()
    
    def get_env_config(self) -> Dict[str, Any]:
        stage_config = self.get_config()
        env_cfg = {
            'env': {
                'initial_T': stage_config['initial_temp'],
                'temp_noise_std': stage_config['temp_noise_std'],
            },
            'plant': {
                'initial_moisture': stage_config['initial_moisture'],
                'moisture_noise_std': stage_config['moisture_noise_std'],
            },
            'disturbance_prob': stage_config['disturbance_prob'],
            'episode_length_days': stage_config['episode_length_days']
        }
        return env_cfg
    
    def should_apply_disturbance(self) -> bool:
        stage_config = self.get_config()
        return np.random.random() < stage_config['disturbance_prob']
    
    def get_disturbance(self) -> Dict[str, Any]:
        disturbance_type = np.random.choice([
            'temp_spike', 'temp_drop', 'moisture_spike', 'moisture_drop',
            'power_outage', 'sensor_failure'
        ])
        disturbances = {
            'temp_spike': {'temp_offset': +8.0, 'duration': 2},
            'temp_drop': {'temp_offset': -8.0, 'duration': 2},
            'moisture_spike': {'moisture_offset': +0.2, 'duration': 1},
            'moisture_drop': {'moisture_offset': -0.2, 'duration': 1},
            'power_outage': {'fan_disabled': True, 'heater_disabled': True, 'duration': 3},
            'sensor_failure': {'sensor_noise_std': 0.5, 'duration': 5},
        }
        return {
            'type': disturbance_type,
            **disturbances[disturbance_type]
        }

class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env, curriculum: CurriculumScheduler):
        super().__init__(env)
        self.curriculum = curriculum
        self.disturbance_active = None
        self.disturbance_remaining = 0
    
    def reset(self, **kwargs):
        curriculum_cfg = self.curriculum.get_env_config()
        options = kwargs.get('options', {})
        options['curriculum'] = curriculum_cfg
        kwargs['options'] = options
        
        self.disturbance_active = None
        self.disturbance_remaining = 0
        
        return self.env.reset(**kwargs)
    
    def step(self, action):
        if self.disturbance_remaining <= 0 and self.curriculum.should_apply_disturbance():
            self.disturbance_active = self.curriculum.get_disturbance()
            self.disturbance_remaining = self.disturbance_active.get('duration', 1)
        
        if self.disturbance_active and self.disturbance_remaining > 0:
            if 'fan_disabled' in self.disturbance_active and self.disturbance_active['fan_disabled']:
                action[1] = 0.0
            if 'heater_disabled' in self.disturbance_active and self.disturbance_active['heater_disabled']:
                action[3] = 0.0
            self.disturbance_remaining -= 1
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        info['curriculum_stage'] = self.curriculum.current_stage
        if self.disturbance_active:
            info['disturbance'] = self.disturbance_active['type']
        
        return obs, reward, terminated, truncated, info