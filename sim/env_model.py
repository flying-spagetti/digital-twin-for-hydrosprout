# sim/env_model.py
"""
EnvironmentModel
----------------
Simulates thermal dynamics, light availability, and evaporation inside the microgreens station.

This model acts as the environmental layer of the digital twin.
"""

import numpy as np

class EnvironmentModel:
    def __init__(self, cfg=None):
        cfg = cfg or {}

        # Core environment params
        self.dt = cfg.get('dt', 1.0)  # timestep (hours)
        self.ambient = cfg.get('ambient', 23.0)  # ambient temp (°C)
        self.T = cfg.get('initial_T', 25.0)      # internal temperature (°C)
        self.C_air = cfg.get('C_air', 1.0)       # heat capacity constant
        self.U_loss = cfg.get('U_loss', 0.1)     # passive heat loss coeff
        self.RH = cfg.get('initial_RH', 60.0)    # relative humidity (%)

        # Shield behavior
        self.shield_factor = cfg.get('shield_factor', 0.6)  # how much shield blocks light

    # ------------------------------------------------------------------
    def solar_input(self, hour_of_day):
        """Simple bell-shaped solar curve peaking at noon."""
        peak = 1.0
        t = (hour_of_day - 12.0)
        val = peak * (1 - (t / 8.0) ** 2)
        return float(max(0.0, val))

    # ------------------------------------------------------------------
    def step(self, hour, shield_pos=0.0, heater_power=0.0, fan_on=False):
        """
        Updates temperature, computes light and evaporation.

        hour: hour of day (0–23)
        shield_pos: 0=open, 1=closed
        heater_power: normalized 0–1
        fan_on: True/False
        """
        # Sunlight reduced by shield
        Q_sun = self.solar_input(hour) * (1.0 - shield_pos * self.shield_factor)

        # Heater adds heat
        Q_heater = heater_power * 1.0

        # Fan increases cooling and reduces humidity
        fan_loss = 0.2 if fan_on else 0.0
        fan_humidity_reduction = 5.0 if fan_on else 0.0  # fan reduces RH

        # Thermal ODE (discretized)
        dT = (
            Q_sun + Q_heater
            - (self.U_loss + fan_loss) * (self.T - self.ambient)
        ) / max(0.1, self.C_air)

        self.T = float(self.T + dT * self.dt)

        # Humidity dynamics (simplified: increases with temp, decreases with fan)
        # Base RH decreases slightly with temperature increase (more water vapor capacity)
        dRH = -0.5 * (self.T - 20.0) / 20.0 - fan_humidity_reduction
        self.RH = float(np.clip(self.RH + dRH * self.dt, 30.0, 90.0))

        # Light reaching plants
        L = float(np.clip(Q_sun * (1 - 0.3 * shield_pos), 0.0, 1.0))

        # Evaporation approx (increases with temp)
        evap_factor = float(np.clip(0.01 * (1 + (self.T - 20) / 20.0), 0.0, 0.2))

        return {
            "T": self.T,
            "RH": self.RH,
            "L": L,
            "evap": evap_factor
        }
