# rl/gym_env.py
"""
DigitalTwinEnv
--------------
A Gymnasium environment that wraps together:
- PlantModel
- EnvironmentModel
- HardwareModel
- SensorModel

This environment provides a unified RL interface with:
    observation: sensor readings
    action: water, fan, shield movement, heater

The environment simulates 14 days with 1-hour timesteps.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from sim.plant import PlantModel
from sim.env_model import EnvironmentModel
from sim.hardware import HardwareModel
from sim.sensors import SensorModel


class DigitalTwinEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or {}

        # Sub-models
        self.plant = PlantModel(self.cfg.get('plant', {}))
        self.env = EnvironmentModel(self.cfg.get('env', {}))
        self.hw = HardwareModel(self.cfg.get('hardware', {}))
        self.sensors = SensorModel(self.cfg.get('sensors', {}))

        # Observation space (normalized values)
        # [canopy, moisture, nutrient, mold, temp_scaled, lux, shield_pos]
        low = np.array([0.0] * 7, dtype=np.float32)
        high = np.array([1.0] * 7, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Action space: [water, fan, shield_delta, heater]
        self.action_space = spaces.Box(
            low=np.array([0, 0, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        self.hour = 6
        self.step_count = 0
        self.max_steps = 24 * 14

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.plant = PlantModel(self.cfg.get('plant', {}))
        self.env = EnvironmentModel(self.cfg.get('env', {}))
        self.hw = HardwareModel(self.cfg.get('hardware', {}))
        self.sensors = SensorModel(self.cfg.get('sensors', {}))

        self.hour = 6
        self.step_count = 0

        obs=self._get_obs()
        info={}
        return obs, info

    # ------------------------------------------------------------------
    def _env_step_passive(self):
        """Perform environment step to get temp/light/evap values."""
        return self.env.step(
            self.hour,
            shield_pos=self.hw.shield_pos,
            heater_power=0.0,
            fan_on=self.hw.fan_on,
        )

    # ------------------------------------------------------------------
    def _get_obs(self):
        env_s = self._env_step_passive()

        plant_state = {
            'C': self.plant.C,
            'M': self.plant.M,
            'N': self.plant.N,
            'P_mold': self.plant.P_mold,
        }
        hw_state = {
            'shield_pos': self.hw.shield_pos,
            'fan_on': self.hw.fan_on,
        }

        sens = self.sensors.read_all(plant_state, env_s, hw_state)

        obs = np.array(
            [
                sens['canopy'],
                sens['moisture'],
                sens['nutrient'],
                sens['pmold'],
                sens['temp'] / 40.0,  # scale approx
                sens['lux'],
                self.hw.shield_pos,
            ],
            dtype=np.float32,
        )
        return obs

    # ------------------------------------------------------------------
    def step(self, action):
        # Decode action
        water = float(action[0])
        fan = int(round(float(action[1])))
        shield_delta = float(action[2])
        heater = float(action[3])

        # Hardware step
        hw_out = self.hw.step(
            {
                'water': water,
                'fan': fan,
                'shield': shield_delta,
                'heater': heater,
            }
        )

        # Environment step (active)
        env_out = self.env.step(
            self.hour,
            shield_pos=self.hw.shield_pos,
            heater_power=hw_out['heater_power'],
            fan_on=self.hw.fan_on,
        )

        # Plant update
        plant_out = self.plant.step(
            env_out['L'],
            env_out['T'],
            env_out['evap'],
            water_input=hw_out['delivered_water'],
            nutrient_input=0.0
        )

        # Next observation
        obs = self._get_obs()

        # Reward
        energy_cost = self.hw.energy
        reward = (
            self.plant.C * 2.0
            - self.plant.P_mold * 5.0
            - 0.01 * energy_cost
            - abs(self.plant.M - 0.45)  # moisture optimal penalty
        )

        # Time update
        self.step_count += 1
        self.hour = (self.hour + 1) % 24

        #done = self.step_count >= self.max_steps
        terminated=False
        truncated=self.step_count >= self.max_steps #Episode ends because of time limit
        info = {
            'plant': plant_out,
            'env': env_out,
            'hw': hw_out,
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        print(
            f"Hour {self.hour} | Canopy {self.plant.C:.3f} | Moist {self.plant.M:.3f} | Temp {self.env.T:.2f}"
        )
