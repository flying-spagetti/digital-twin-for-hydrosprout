# sim/plant.py
"""
PlantModel
-----------
This module defines the plant growth simulation for the digital twin.
It models canopy growth, moisture, nutrient uptake, and mold probability.

All values are normalized to 0–1 unless otherwise noted.
"""

import math
import numpy as np

class PlantModel:
    def __init__(self, cfg=None):
        cfg = cfg or {}

        # Growth parameters
        self.r_base = cfg.get('r_base', 0.02)
        self.k_l = cfg.get('k_l', 0.2)
        self.M_opt = cfg.get('M_opt', 0.45)
        self.sigma_m = cfg.get('sigma_m', 0.2)
        self.T_opt = cfg.get('T_opt', 20.0)
        self.sigma_t = cfg.get('sigma_t', 6.0)
        self.k_n = cfg.get('k_n', 0.2)
        self.decay_rate = cfg.get('decay_rate', 0.001)

        # Initial states
        self.C = 0.02   # canopy fraction (0–1)
        self.M = 0.45   # moisture (0–1)
        self.N = 0.60   # nutrient availability (0–1)
        self.P_mold = 0.0  # mold probability (0–1)

    # Response curves ------------------------------------------------------
    def phi(self, L):
        """Light response: saturating curve."""
        return L / (L + self.k_l)

    def psi(self, M):
        """Moisture response: bell-shaped around M_opt."""
        return math.exp(-((M - self.M_opt) / self.sigma_m) ** 2)

    def tau(self, T):
        """Temperature response: Gaussian centered on T_opt."""
        return math.exp(-((T - self.T_opt) / self.sigma_t) ** 2)

    def nu(self, N):
        """Nutrient response: saturating curve."""
        return N / (N + self.k_n)

    # Step update ----------------------------------------------------------
    def step(self, light, temp, evap_factor, water_input=0.0, nutrient_input=0.0, dt=1.0):
        """
        Updates the plant state by dt hours.
        Inputs:
            light: normalized (0–1)
            temp: degrees C
            evap_factor: moisture loss rate from environment
            water_input: normalized amount delivered this step
        """
        # Growth
        growth = (
            self.r_base
            * self.phi(light)
            * self.psi(self.M)
            * self.tau(temp)
            * self.nu(self.N)
            * dt
        )

        decay = self.decay_rate * self.C * dt
        self.C = float(np.clip(self.C + growth - decay, 0.0, 1.0))

        # Moisture
        alpha_w = 0.30  # water absorption factor
        percolation = 0.01
        self.M = float(
            np.clip(
                self.M * (1 - evap_factor)
                + alpha_w * water_input
                - percolation * self.M,
                0.0,
                1.0,
            )
        )

        # Nutrients
        uptake = 0.1 * growth
        self.N = float(np.clip(self.N - uptake + nutrient_input, 0.0, 1.0))

        # Mold probability (hazard accumulation)
        a1, a2, a3 = 0.6, 0.4, 0.5
        hum_term = 0.5  # placeholder; real env should provide humidity
        H = (
            a1 * (1 / (1 + math.exp(-((self.M - 0.7) / 0.05))))
            + a2 * (1 / (1 + math.exp(-((hum_term - 0.7) / 0.05))))
            - a3 * 0.5
        )
        self.P_mold = float(np.clip(self.P_mold + H * dt * 0.01, 0.0, 1.0))

        return {
            "C": self.C,
            "M": self.M,
            "N": self.N,
            "P_mold": self.P_mold,
        }

    # Stage ---------------------------------------------------------------
    def stage(self):
        """Returns growth stage index: 0=germ, 1=sprout, 2=ready."""
        if self.C >= 0.5:
            return 2
        elif self.C >= 0.15:
            return 1
        return 0

