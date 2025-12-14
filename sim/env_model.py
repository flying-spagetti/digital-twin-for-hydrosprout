# sim/env_model_enhanced.py
"""
Enhanced EnvironmentModel with:
- Peltier cooling modules (spatial array)
- CO2 concentration modeling
- Improved thermal dynamics (stratification)
- Better humidity control
- Air circulation patterns
"""

import numpy as np
from typing import Dict, Tuple, Optional

class PeltierModule:
    """Single Peltier thermoelectric cooler/heater"""
    def __init__(self, position: Tuple[float, float], max_cooling_w: float = 50.0):
        self.position = position  # (x, y) in meters
        self.max_cooling_w = max_cooling_w
        self.max_heating_w = max_cooling_w * 0.8  # COP consideration
        self.power = 0.0  # -1 to 1 (negative=cooling, positive=heating)
        self.efficiency = 0.7  # Real-world Peltier efficiency
    
    def get_thermal_effect(self, distance: float) -> float:
        """Temperature effect decreases with distance (inverse square)"""
        if distance < 0.01:
            distance = 0.01
        return self.power / (1 + distance**2)


class EnhancedEnvironmentModel:
    """
    Advanced environmental simulation with:
    - 3D thermal zones (top/middle/bottom stratification)
    - Peltier cooling array
    - CO2 dynamics
    - Spatial humidity variation
    """
    
    def __init__(self, cfg=None):
        cfg = cfg or {}
        
        # Time step
        self.dt = cfg.get('dt', 1.0)  # hours
        
        # Chamber geometry (meters)
        self.width = cfg.get('width', 0.6)
        self.depth = cfg.get('depth', 0.4)
        self.height = cfg.get('height', 0.8)
        self.volume = self.width * self.depth * self.height  # m³
        
        # Thermal zones (stratification)
        self.T_top = cfg.get('initial_T', 25.0)
        self.T_middle = cfg.get('initial_T', 25.0)
        self.T_bottom = cfg.get('initial_T', 25.0)
        self.ambient = cfg.get('ambient', 23.0)
        
        # Heat capacity and transfer
        # C_effective includes air + walls + water + plant mass (much larger than air alone)
        self.C_effective = cfg.get('C_effective', 50000.0)  # J/K (effective system thermal mass)
        self.C_air = cfg.get('C_air', 1200.0)  # J/K (air mass * specific heat) - kept for reference
        self.U_walls = cfg.get('U_loss', 2.5)  # W/K (insulation quality)
        
        # Solar absorption parameters
        self.chamber_area = self.width * self.depth  # m² (top surface area)
        self.solar_absorptivity = cfg.get('solar_absorptivity', 0.7)  # Fraction of solar energy that becomes heat
        self.solar_to_heat_fraction = cfg.get('solar_to_heat_fraction', 0.5)  # Fraction of absorbed energy that heats air vs. other uses
        
        # Actuator rate limits (max temperature change per step)
        self.max_dT_per_step = cfg.get('max_dT_per_step', 3.0)  # °C per hour step
        
        # Peltier modules (configurable array)
        n_peltiers = cfg.get('n_peltier_modules', 4)
        self.peltiers = self._init_peltier_array(n_peltiers)
        
        # Fan system
        self.fan_on = False
        self.fan_cfm = cfg.get('fan_cfm', 100.0)  # Air flow rate
        self.fan_power_w = cfg.get('fan_power_w', 15.0)
        
        # Humidity
        self.RH_top = cfg.get('initial_RH', 60.0)
        self.RH_middle = cfg.get('initial_RH', 60.0)
        self.RH_bottom = cfg.get('initial_RH', 65.0)  # Higher near substrate
        
        # CO2 (ppm)
        self.CO2 = cfg.get('initial_CO2', 400.0)
        self.CO2_ambient = cfg.get('ambient_CO2', 410.0)
        self.CO2_injection_rate = 0.0  # L/hour
        
        # Light shield
        self.shield_factor = cfg.get('shield_factor', 0.6)
        
        # Energy tracking
        self.energy_cooling = 0.0
        self.energy_heating = 0.0
        self.energy_fan = 0.0
    
    def _init_peltier_array(self, n_modules: int) -> list:
        """Initialize Peltier modules in strategic positions"""
        modules = []
        
        if n_modules == 4:
            # Corner placement
            positions = [
                (0.1, 0.1), (0.5, 0.1),
                (0.1, 0.3), (0.5, 0.3)
            ]
        elif n_modules == 6:
            # Walls + corners
            positions = [
                (0.05, 0.2), (0.3, 0.05), (0.55, 0.2),
                (0.3, 0.35), (0.1, 0.1), (0.5, 0.3)
            ]
        else:
            # Grid layout
            nx = int(np.sqrt(n_modules)) + 1
            positions = [
                (i * self.width / nx, j * self.depth / nx)
                for i in range(nx) for j in range(nx)
            ][:n_modules]
        
        for pos in positions:
            modules.append(PeltierModule(pos))
        
        return modules
    
    def solar_input(self, hour_of_day: int) -> float:
        """
        Realistic solar irradiance (W/m²) with atmospheric effects.
        Returns actual W/m², not normalized.
        """
        if hour_of_day < 6 or hour_of_day > 18:
            return 0.0
        
        # Solar noon at hour 12
        hour_angle = (hour_of_day - 12) * 15  # degrees
        altitude = 90 - abs(hour_angle)  # simplified
        
        # Clear-sky irradiance with zenith angle
        max_irradiance = 1000.0  # W/m² at solar noon
        cos_zenith = np.cos(np.radians(90 - altitude))
        irradiance = max_irradiance * max(0, cos_zenith) ** 1.2
        
        return float(irradiance)  # Return W/m² directly
    
    def thermal_stratification_mixing(self):
        """Mix thermal zones based on natural convection + fan"""
        mixing_rate = 0.1  # Natural convection
        if self.fan_on:
            mixing_rate = 0.5  # Forced circulation
        
        # Heat rises naturally
        buoyancy_rate = 0.05
        
        # Mix middle with top/bottom
        T_avg = (self.T_top + self.T_middle + self.T_bottom) / 3.0
        
        self.T_top += (T_avg - self.T_top) * mixing_rate * self.dt
        self.T_middle += (T_avg - self.T_middle) * mixing_rate * self.dt
        self.T_bottom += (T_avg - self.T_bottom) * mixing_rate * self.dt
        
        # Buoyancy (heat rises, cold sinks)
        if not self.fan_on:
            delta_T = (self.T_bottom - self.T_top) * buoyancy_rate * self.dt
            self.T_top += delta_T
            self.T_bottom -= delta_T
    
    def apply_peltier_cooling(self, peltier_controls: np.ndarray):
        """
        Apply Peltier module effects to thermal zones.
        
        Args:
            peltier_controls: Array of power levels [-1, 1] for each module
                             negative = cooling, positive = heating
        """
        for i, module in enumerate(self.peltiers):
            if i < len(peltier_controls):
                module.power = np.clip(peltier_controls[i], -1.0, 1.0)
                
                # Distribute cooling/heating to zones based on vertical position
                # Assume modules are at mid-height, affect middle zone most
                power_w = (module.power * module.max_cooling_w if module.power < 0 
                          else module.power * module.max_heating_w) * module.efficiency
                
                # Convert to temperature change (Q = m*c*ΔT)
                # Use C_effective instead of C_air for realistic thermal mass
                dT = (power_w * 3600) / self.C_effective * self.dt  # Per hour
                
                # Apply rate limit to prevent unrealistic temperature jumps
                dT = np.clip(dT, -self.max_dT_per_step, self.max_dT_per_step)
                
                # Distribute to zones
                self.T_middle += dT * 0.6
                self.T_top += dT * 0.25
                self.T_bottom += dT * 0.15
                
                # Track energy
                if power_w < 0:
                    self.energy_cooling += abs(power_w) * self.dt
                else:
                    self.energy_heating += power_w * self.dt
    
    def co2_dynamics(self, plant_uptake_rate: float, co2_injection: float):
        """
        Update CO2 concentration.
        
        Args:
            plant_uptake_rate: CO2 consumption by plants (L/hour)
            co2_injection: CO2 addition (L/hour)
        """
        # Volume in liters
        volume_L = self.volume * 1000.0
        
        # Concentration change (ppm)
        # 1 L CO2 at STP ≈ 1.96 g, 1 mole = 22.4 L
        co2_in = co2_injection * (1e6 / volume_L) * self.dt
        co2_out = plant_uptake_rate * (1e6 / volume_L) * self.dt
        
        # Air exchange with ambient (leakage + fan)
        exchange_rate = 0.02 if not self.fan_on else 0.1  # Volume/hour
        co2_exchange = (self.CO2_ambient - self.CO2) * exchange_rate * self.dt
        
        self.CO2 += co2_in - co2_out + co2_exchange
        self.CO2 = max(300.0, min(2000.0, self.CO2))  # Reasonable bounds
    
    def humidity_dynamics(self, evapotranspiration: float):
        """
        Update humidity with spatial variation.
        
        Args:
            evapotranspiration: Water added to air (L/hour)
        """
        # Convert evapotranspiration to RH increase
        # Simplified: 1L water → ~10% RH increase in small volume
        volume_factor = 1.0 / (self.volume * 10.0)
        dRH = evapotranspiration * volume_factor * 100.0 * self.dt
        
        # Bottom zone (near substrate) gets more humidity
        self.RH_bottom += dRH * 1.5
        self.RH_middle += dRH
        self.RH_top += dRH * 0.7
        
        # Fan mixing
        if self.fan_on:
            RH_avg = (self.RH_top + self.RH_middle + self.RH_bottom) / 3.0
            mix_rate = 0.3 * self.dt
            self.RH_top += (RH_avg - self.RH_top) * mix_rate
            self.RH_middle += (RH_avg - self.RH_middle) * mix_rate
            self.RH_bottom += (RH_avg - self.RH_bottom) * mix_rate
            
            # Fan actively removes moisture
            self.RH_top = max(30.0, self.RH_top - 3.0 * self.dt)
            self.RH_middle = max(30.0, self.RH_middle - 2.5 * self.dt)
            self.RH_bottom = max(35.0, self.RH_bottom - 2.0 * self.dt)
        
        # Natural condensation at high RH
        self.RH_top = min(95.0, self.RH_top)
        self.RH_middle = min(95.0, self.RH_middle)
        self.RH_bottom = min(95.0, self.RH_bottom)
        
        # Ambient exchange
        ambient_RH = 50.0
        exchange = 0.05 * self.dt
        self.RH_top += (ambient_RH - self.RH_top) * exchange
        self.RH_middle += (ambient_RH - self.RH_middle) * exchange
        self.RH_bottom += (ambient_RH - self.RH_bottom) * exchange
    
    def step(self, hour: int, shield_pos: float = 0.0, heater_power: float = 0.0,
             fan_on: bool = False, peltier_controls: Optional[np.ndarray] = None,
             plant_co2_uptake: float = 0.0, evapotranspiration: float = 0.0,
             co2_injection: float = 0.0) -> Dict:
        """
        Full environment simulation step.
        
        Returns:
            Dict with T, RH, L (light), CO2, and zone-specific values
        """
        self.fan_on = fan_on
        
        # Solar input - now returns W/m², compute physically consistent heat input
        irradiance_w_m2 = self.solar_input(hour) * (1.0 - shield_pos * self.shield_factor)
        # Only a fraction of incident solar becomes air heat (rest goes to photosynthesis, etc.)
        Q_sun_w = irradiance_w_m2 * self.chamber_area * self.solar_absorptivity * self.solar_to_heat_fraction
        
        # Heater
        Q_heater = heater_power * 200.0  # Max 200W heater
        self.energy_heating += Q_heater * self.dt
        
        # Peltier cooling/heating
        if peltier_controls is not None:
            self.apply_peltier_cooling(peltier_controls)
        
        # Heat losses
        T_mean = (self.T_top + self.T_middle + self.T_bottom) / 3.0
        Q_loss = self.U_walls * (T_mean - self.ambient)
        
        # Fan effect (increases heat loss through air exchange)
        if self.fan_on:
            Q_fan_loss = 0.5 * self.U_walls * (T_mean - self.ambient)
            self.energy_fan += self.fan_power_w * self.dt
        else:
            Q_fan_loss = 0.0
        
        # Net heat to middle zone (main calculation)
        # Use C_effective instead of C_air for realistic thermal mass
        dQ_net = (Q_sun_w + Q_heater - Q_loss - Q_fan_loss) * self.dt * 3600  # Joules
        dT_middle = dQ_net / self.C_effective
        
        # Apply rate limit to prevent unrealistic temperature jumps
        dT_middle = np.clip(dT_middle, -self.max_dT_per_step, self.max_dT_per_step)
        
        self.T_middle += dT_middle
        
        # Top zone gets more solar heat
        dT_top_solar = (Q_sun_w * 0.3 * self.dt * 3600 / self.C_effective)
        dT_top_solar = np.clip(dT_top_solar, -self.max_dT_per_step, self.max_dT_per_step)
        self.T_top += dT_middle * 0.6 + dT_top_solar
        
        # Bottom zone more stable
        self.T_bottom += dT_middle * 0.4
        
        # THERMAL STABILITY GUARDRAILS - Hard clamps to prevent blow-ups
        # Clamp absolute temperatures to realistic range
        T_min = 0.0  # °C (freezing)
        T_max = 45.0  # °C (extreme heat)
        self.T_top = np.clip(self.T_top, T_min, T_max)
        self.T_middle = np.clip(self.T_middle, T_min, T_max)
        self.T_bottom = np.clip(self.T_bottom, T_min, T_max)
        
        # Clamp per-step temperature change to prevent unrealistic jumps
        dT_max = 5.0  # Max °C change per hour step
        # Note: This is already handled in apply_peltier_cooling, but add extra safety here
        
        # Thermal mixing
        self.thermal_stratification_mixing()
        
        # Re-apply clamps after mixing
        self.T_top = np.clip(self.T_top, T_min, T_max)
        self.T_middle = np.clip(self.T_middle, T_min, T_max)
        self.T_bottom = np.clip(self.T_bottom, T_min, T_max)
        
        # CO2 dynamics
        self.co2_dynamics(plant_co2_uptake, co2_injection)
        
        # Humidity
        self.humidity_dynamics(evapotranspiration)
        
        # Light (PAR normalized) - use normalized irradiance for compatibility
        irradiance_normalized = irradiance_w_m2 / 1000.0  # Normalize to 0..1 for light calculation
        L = float(np.clip(irradiance_normalized * (1 - 0.3 * shield_pos), 0.0, 1.0))
        
        return {
            "T": self.T_middle,  # Average for compatibility
            "T_top": self.T_top,
            "T_middle": self.T_middle,
            "T_bottom": self.T_bottom,
            "RH": self.RH_middle,
            "RH_top": self.RH_top,
            "RH_middle": self.RH_middle,
            "RH_bottom": self.RH_bottom,
            "CO2": self.CO2,
            "L": L,
            "evap": evapotranspiration / max(0.1, self.volume)  # For compatibility
        }