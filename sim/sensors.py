# sim/sensors.py
"""
SensorModel
-----------
Simulates readings from sensors with configurable noise, drift, and occasional spikes.

Provides a `read_all` method that returns a dictionary of sensor values.
"""

import random

class SensorModel:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        # noise standard deviations
        self.sigma_moist = cfg.get('sigma_moist', 0.02)
        self.sigma_temp = cfg.get('sigma_temp', 0.2)
        self.sigma_lux = cfg.get('sigma_lux', 0.05)
        # sensor drift terms
        self.drift_moist = cfg.get('drift_moist', 0.0)
        self.drift_temp = cfg.get('drift_temp', 0.0)
        self.drift_lux = cfg.get('drift_lux', 0.0)
        # spike probability
        self.spike_prob = cfg.get('spike_prob', 0.001)

    def _maybe_spike(self, base, sigma):
        if random.random() < self.spike_prob:
            return base + random.gauss(0, 4 * sigma)
        return base

    def read_all(self, plant_state, env_state, hardware_state):
        """
        plant_state: dict with keys 'C','M','N','P_mold'
        env_state: dict with keys 'T','L','evap'
        hardware_state: dict with keys 'shield_pos','fan_on','delivered_water'
        """
        C = plant_state.get('C', 0.0)
        M = plant_state.get('M', 0.0)
        N = plant_state.get('N', 0.0)
        Pm = plant_state.get('P_mold', 0.0)
        T = env_state.get('T', 20.0)
        L = env_state.get('L', 0.0)

        # add drift and gaussian noise
        moist_noisy = M + random.gauss(0, self.sigma_moist) + self.drift_moist
        temp_noisy = T + random.gauss(0, self.sigma_temp) + self.drift_temp
        lux_noisy = L * (1.0 + random.gauss(0, self.sigma_lux) + self.drift_lux)

        # occasional spike
        moist_noisy = self._maybe_spike(moist_noisy, self.sigma_moist)
        temp_noisy = self._maybe_spike(temp_noisy, self.sigma_temp)
        lux_noisy = self._maybe_spike(lux_noisy, self.sigma_lux)

        # clip sensible ranges
        moist_s = float(max(0.0, min(1.0, moist_noisy)))
        temp_s = float(temp_noisy)
        lux_s = float(max(0.0, min(1.0, lux_noisy)))

        return {
            'canopy': float(C),
            'moisture': moist_s,
            'nutrient': float(N),
            'pmold': float(Pm),
            'temp': temp_s,
            'lux': lux_s
        }
