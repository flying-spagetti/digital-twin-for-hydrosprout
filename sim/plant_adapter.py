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
        # Handle case where soil_theta might not be initialized yet
        if hasattr(self.plant, 'soil_theta'):
            return float(self.plant.soil_theta)
        else:
            # Fallback: initialize if missing
            self.plant.soil_theta = 0.35
            return 0.35
    
    @property
    def N(self):
        """Nutrient availability (0..1) - computed from soil macronutrients"""
        # PlantStructural uses SoilModelExtended which has separate N, P, K
        # Compute a combined nutrient availability metric
        if hasattr(self.plant, 'soil') and hasattr(self.plant.soil, 'soil_N'):
            # Weighted average: N is most important, then P, then K
            soil_N = float(self.plant.soil.soil_N)
            soil_P = float(self.plant.soil.soil_P) if hasattr(self.plant.soil, 'soil_P') else 0.0
            soil_K = float(self.plant.soil.soil_K) if hasattr(self.plant.soil, 'soil_K') else 0.0
            # Weighted average: N=0.5, P=0.3, K=0.2
            combined = 0.5 * soil_N + 0.3 * soil_P + 0.2 * soil_K
            return float(np.clip(combined, 0.0, 1.0))
        else:
            # Fallback if soil model not available
            return 0.6
    
    @property
    def P_mold(self):
        """Mold probability (0..1) - average across all plants"""
        return float(np.mean(self.plant.mold_prob))
    
    def is_dead(self, temp=None):
        """
        Check if plants are dead based on:
        - Biomass too low (below minimum viable threshold)
        - Extreme temperature (cooked/frozen)
        - Extreme moisture (dried out or waterlogged)
        
        Args:
            temp: Current temperature (°C) for death check
        
        Returns:
            bool: True if plants are dead
        """
        if not hasattr(self.plant, 'B_leaf'):
            return False
        
        # Check biomass death: if total biomass per plant is below minimum viable
        total_biomass = np.sum(self.plant.B_leaf + self.plant.B_root + self.plant.B_stem)
        biomass_per_plant = total_biomass / max(1, self.plant.n)
        min_viable_biomass = 0.005  # 5mg per plant - below this, plant is dead
        if biomass_per_plant < min_viable_biomass:
            return True
        
        # Check temperature death (cooked or frozen)
        if temp is not None:
            # Plants die if temp > 40°C (cooked) or < 5°C (frozen)
            if temp > 40.0 or temp < 5.0:
                return True
        
        # Check moisture death (dried out or waterlogged)
        # Soil moisture too low (< 0.1) = dried out, too high (> 0.95) = waterlogged/root rot
        if hasattr(self.plant, 'soil_theta'):
            if self.plant.soil_theta < 0.1 or self.plant.soil_theta > 0.95:
                return True
        
        return False
    
    def get_death_reason(self, temp=None):
        """
        Get reason for plant death (for diagnostics).
        
        Returns:
            str: Reason for death, or None if not dead
        """
        if not hasattr(self.plant, 'B_leaf'):
            return None
        
        total_biomass = np.sum(self.plant.B_leaf + self.plant.B_root + self.plant.B_stem)
        biomass_per_plant = total_biomass / max(1, self.plant.n)
        
        if biomass_per_plant < 0.005:
            return "biomass_too_low"
        
        if temp is not None:
            if temp > 40.0:
                return "cooked"
            if temp < 5.0:
                return "frozen"
        
        if hasattr(self.plant, 'soil_theta'):
            if self.plant.soil_theta < 0.1:
                return "dried_out"
            if self.plant.soil_theta > 0.95:
                return "waterlogged"
        
        return None
    
    def step(self, light, temp, evap_factor, water_input=0.0, nutrient_input=0.0, dt=1.0, env_state=None):
        """
        Step function matching PlantModel interface.
        
        Args:
            light: normalized light (0..1) - will be used as I_norm
            temp: temperature (°C)
            evap_factor: evaporation factor (not directly used, but affects RH)
            water_input: normalized water input (0..1) - converted to liters
            nutrient_input: normalized nutrient input (0..1) OR dict with {'N':val, 'P':val, 'K':val, 'micro':{...}, 'chelated':bool}
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
        
        # Convert nutrient_input to dict format if it's a scalar
        nutrient_dose = None
        if nutrient_input:
            if isinstance(nutrient_input, dict):
                # Already in dict format (for SoilModelExtended)
                nutrient_dose = nutrient_input
            else:
                # Convert scalar to simple N dose (backward compatibility)
                nutrient_dose = {
                    'N': float(nutrient_input),
                    'P': 0.0,
                    'K': 0.0,
                    'micro': None,
                    'chelated': False
                }
        
        # Step PlantStructural
        diagnostics = self.plant.step(
            dt=dt,
            env_state=env_state_ps,
            water_liters=water_liters,
            nutrient_dose=nutrient_dose
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

