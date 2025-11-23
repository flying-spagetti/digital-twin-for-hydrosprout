# sim/plant_adapter.py
"""
Adapter wrapper for PlantStructural to match PlantModel interface.

This allows PlantStructural to be used as a drop-in replacement for PlantModel
in the gym_env and main.py.
"""

import numpy as np
from sim.plant_SFPM import PlantStructural


class PlantStructuralAdapter:
    """
    Adapter that wraps PlantStructural to match the PlantModel interface.
    
    Provides:
    - Properties: C (canopy), M (moisture), N (nutrient), P_mold (mold probability)
    - step() method with same signature as PlantModel
    """
    
    def __init__(self, cfg=None):
        """
        Initialize PlantStructural with config.
        
        cfg: dict with plant parameters (can override PlantStructural defaults)
        """
        cfg = cfg or {}
        
        # Extract PlantStructural-specific params
        rows = cfg.get('rows', 8)
        cols = cfg.get('cols', 4)
        tray_area = cfg.get('tray_area_m2', 0.06)
        seed = cfg.get('seed', None)
        
        # Create PlantStructural instance
        self.plant = PlantStructural(
            rows=rows,
            cols=cols,
            tray_area_m2=tray_area,
            params=cfg.get('params', None),
            seed=seed
        )
        
        # Conversion factor: normalized water (0..1) to liters
        # Assuming 0.0 = no water, 1.0 = full watering (~0.5 liters for microgreens tray)
        self.water_scale = cfg.get('water_scale', 0.5)
    
    @property
    def C(self):
        """Canopy cover fraction (0..1)"""
        return float(self.plant.canopy_cover)
    
    @property
    def M(self):
        """Soil moisture (0..1) - normalized from soil_theta"""
        # soil_theta is already 0..1, but we might want to normalize to field capacity
        # For now, return soil_theta directly
        return float(self.plant.soil_theta)
    
    @property
    def N(self):
        """Nutrient availability (0..1)"""
        return float(self.plant.soil_nutrient)
    
    @property
    def P_mold(self):
        """Mold probability (0..1) - average across all plants"""
        return float(np.mean(self.plant.mold_prob))
    
    def step(self, light, temp, evap_factor, water_input=0.0, nutrient_input=0.0, dt=1.0, env_state=None):
        """
        Step function matching PlantModel interface.
        
        Args:
            light: normalized light (0..1) - will be used as I_norm
            temp: temperature (°C)
            evap_factor: evaporation factor (not directly used, but affects RH)
            water_input: normalized water input (0..1) - converted to liters
            nutrient_input: normalized nutrient input (0..1)
            dt: timestep in hours
            env_state: optional dict with 'RH' key for relative humidity (%)
        
        Returns:
            dict with C, M, N, P_mold (matching PlantModel output)
        """
        # Convert normalized water to liters
        water_liters = water_input * self.water_scale
        
        # Get RH from env_state if provided, otherwise estimate from evap_factor
        if env_state and 'RH' in env_state:
            RH = env_state['RH']
        else:
            # Estimate RH from evap_factor and temperature
            # Higher evap_factor -> lower RH (more evaporation means drier air)
            estimated_RH = max(30.0, min(90.0, 70.0 - evap_factor * 100.0))
            RH = estimated_RH
        
        env_state_ps = {
            'T': temp,
            'RH': RH,
            'I_norm': light
        }
        
        # Step PlantStructural
        diagnostics = self.plant.step(
            dt=dt,
            env_state=env_state_ps,
            water_liters=water_liters,
            nutrient_dose=nutrient_input
        )
        
        # Return in PlantModel format
        return {
            'C': self.C,
            'M': self.M,
            'N': self.N,
            'P_mold': self.P_mold,
            'diagnostics': diagnostics  # Include full diagnostics for advanced use
        }
    
    def stage(self):
        """Returns growth stage index: 0=germ, 1=sprout, 2=ready."""
        if self.C >= 0.5:
            return 2
        elif self.C >= 0.15:
            return 1
        return 0

