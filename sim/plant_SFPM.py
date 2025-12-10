# sim/plant_fspm.py
"""
Enhanced Functional-Structural Plant Model (FSPM) for Hydroponics
Based on GreenLab architecture and organ-level biomass allocation
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class PlantParameters:
    """Crop-specific physiological parameters"""
    # Photosynthesis parameters (Farquhar model simplified)
    Vcmax: float = 60.0  # Maximum carboxylation rate (μmol CO2/m²/s)
    Jmax: float = 120.0  # Maximum electron transport rate (μmol/m²/s)
    alpha: float = 0.3  # Quantum yield of photosynthesis
    theta: float = 0.7  # Curvature parameter
    
    # Temperature response (Arrhenius)
    T_opt: float = 25.0  # Optimal temperature (°C)
    T_min: float = 10.0  # Minimum temperature for growth
    T_max: float = 35.0  # Maximum temperature for growth
    
    # Respiration
    Q10: float = 2.0  # Temperature coefficient
    Rm_leaf: float = 0.015  # Maintenance respiration coefficient (g/g/day)
    Rm_root: float = 0.01
    Rm_stem: float = 0.005
    
    # Growth and allocation
    SLA: float = 0.025  # Specific Leaf Area (m²/g)
    RGR_max: float = 0.2  # Maximum relative growth rate (1/day)
    k_ext: float = 0.6  # Light extinction coefficient
    
    # Water relations
    transp_coef: float = 0.004  # Transpiration coefficient (L/m²LAI/day at VPD=1kPa)
    water_stress_threshold: float = 0.30  # Moisture below which stress occurs
    
    # Nutrient uptake
    N_uptake_rate: float = 0.02  # N uptake rate (g/g root/day at optimal)
    nutrient_stress_threshold: float = 0.25
    
    # Partitioning coefficients (change with development)
    partition_leaf_early: float = 0.6
    partition_root_early: float = 0.3
    partition_stem_early: float = 0.1
    partition_leaf_late: float = 0.3
    partition_root_late: float = 0.15
    partition_stem_late: float = 0.55


@dataclass
class PlantOrgans:
    """Organ-level state variables (biomass in grams dry weight)"""
    B_leaf: np.ndarray = field(default_factory=lambda: np.array([0.5]))  # Leaf biomass per cohort
    B_stem: np.ndarray = field(default_factory=lambda: np.array([0.2]))  # Stem biomass per cohort
    B_root: np.ndarray = field(default_factory=lambda: np.array([0.3]))  # Root biomass per cohort
    
    # Architectural variables
    LAI: float = 0.0  # Leaf Area Index
    plant_height: float = 0.05  # meters
    root_depth: float = 0.1  # meters
    
    # Physiological states
    NSC: float = 0.5  # Non-structural carbohydrates (g)
    N_content: float = 0.03  # Total N in plant (g)
    
    # Age tracking (thermal time or days)
    thermal_time: float = 0.0
    n_cohorts: int = 1


class HydroponicPlantFSPM:
    """
    Functional-Structural Plant Model for hydroponic systems.
    
    Core principles:
    1. Organ-level resolution (leaves, stems, roots as cohorts)
    2. Source-sink dynamics for biomass allocation
    3. Environmental response functions (light, temp, water, nutrients)
    4. Explicit growth stages (vegetative → reproductive)
    """
    
    def __init__(self, params: Optional[PlantParameters] = None, 
                 initial_biomass: float = 1.0,
                 dt_hours: float = 1.0):
        self.params = params or PlantParameters()
        self.dt = dt_hours / 24.0  # Convert to days
        
        # Initialize organs
        self.organs = PlantOrgans()
        self.organs.B_leaf = np.array([initial_biomass * 0.5])
        self.organs.B_stem = np.array([initial_biomass * 0.2])
        self.organs.B_root = np.array([initial_biomass * 0.3])
        
        # Environmental memory (for integration)
        self.light_integral_daily = 0.0  # DLI tracker
        self.hours_in_day = 0
        
        # Soil/solution state
        self.soil_moisture = 0.5  # Fraction of field capacity
        self.solution_N = 0.5  # Normalized nutrient concentration
        self.solution_EC = 1.5  # Electrical conductivity (dS/m)
        
        # Diagnostics
        self.daily_photosynthesis = 0.0
        self.daily_respiration = 0.0
        self.daily_transpiration = 0.0
        self.stress_factors = {'water': 1.0, 'temp': 1.0, 'nutrient': 1.0}
        
        self._update_lai()
    
    def _update_lai(self):
        """Calculate Leaf Area Index from leaf biomass"""
        total_leaf_biomass = np.sum(self.organs.B_leaf)
        self.organs.LAI = total_leaf_biomass * self.params.SLA
    
    def _temperature_response(self, T: float) -> float:
        """
        Cardinal temperature response function.
        Returns factor in [0, 1] representing temperature suitability.
        """
        T_min = self.params.T_min
        T_opt = self.params.T_opt
        T_max = self.params.T_max
        
        if T <= T_min or T >= T_max:
            return 0.0
        
        if T <= T_opt:
            return (T - T_min) / (T_opt - T_min)
        else:
            return 1.0 - ((T - T_opt) / (T_max - T_opt))
    
    def _water_stress(self, moisture: float) -> float:
        """
        Water stress factor based on soil moisture.
        Returns factor in [0, 1], where 1 = no stress.
        """
        threshold = self.params.water_stress_threshold
        if moisture >= 0.5:
            return 1.0
        elif moisture <= threshold:
            return 0.0
        else:
            # Linear transition
            return (moisture - threshold) / (0.5 - threshold)
    
    def _nutrient_stress(self, nutrient_level: float) -> float:
        """
        Nutrient stress factor.
        Returns factor in [0, 1], where 1 = no stress.
        """
        threshold = self.params.nutrient_stress_threshold
        if nutrient_level >= 0.6:
            return 1.0
        elif nutrient_level <= threshold:
            return 0.1  # Not zero to allow recovery
        else:
            return 0.1 + 0.9 * (nutrient_level - threshold) / (0.6 - threshold)
    
    def photosynthesis(self, PAR: float, T: float, CO2: float = 400.0) -> float:
        """
        Canopy-level photosynthesis using simplified Farquhar model.
        
        Args:
            PAR: Photosynthetically active radiation (μmol/m²/s)
            T: Temperature (°C)
            CO2: CO2 concentration (ppm)
        
        Returns:
            Net photosynthesis rate (g CO2/m²ground/hour)
        """
        if self.organs.LAI < 0.01:
            return 0.0
        
        # Temperature adjustment
        T_factor = self._temperature_response(T)
        if T_factor == 0.0:
            return 0.0
        
        # Light interception (Beer's law)
        I_0 = PAR  # Incident PAR
        k = self.params.k_ext
        LAI = self.organs.LAI
        
        # Absorbed PAR by canopy
        I_abs = I_0 * (1 - np.exp(-k * LAI))
        
        # Rectangular hyperbola for light response
        Vcmax = self.params.Vcmax * T_factor
        alpha = self.params.alpha
        theta = self.params.theta
        
        # Simplified: gross photosynthesis
        Pg = (alpha * I_abs + Vcmax - 
              np.sqrt((alpha * I_abs + Vcmax)**2 - 4 * theta * alpha * I_abs * Vcmax)) / (2 * theta)
        
        # Convert from μmol CO2/m²/s to g CO2/m²/hour
        # 1 μmol CO2 = 44 μg CO2
        Pg_g = Pg * 44e-6 * 3600  # g CO2/m²ground/hour
        
        # Apply stress factors
        water_stress = self._water_stress(self.soil_moisture)
        nutrient_stress = self._nutrient_stress(self.solution_N)
        
        Pg_g *= water_stress * nutrient_stress
        
        self.stress_factors = {
            'water': water_stress,
            'temp': T_factor,
            'nutrient': nutrient_stress
        }
        
        return max(0.0, Pg_g)
    
    def respiration(self, T: float) -> float:
        """
        Maintenance respiration for all organs.
        
        Returns:
            Total respiration rate (g CO2/hour)
        """
        # Q10 temperature correction
        Q10 = self.params.Q10
        T_ref = 20.0
        T_corr = Q10 ** ((T - T_ref) / 10.0)
        
        # Organ-specific respiration
        R_leaf = np.sum(self.organs.B_leaf) * self.params.Rm_leaf * T_corr / 24.0
        R_stem = np.sum(self.organs.B_stem) * self.params.Rm_stem * T_corr / 24.0
        R_root = np.sum(self.organs.B_root) * self.params.Rm_root * T_corr / 24.0
        
        return R_leaf + R_stem + R_root
    
    def transpiration(self, T: float, RH: float, LAI: float = None) -> float:
        """
        Calculate transpiration rate.
        
        Args:
            T: Temperature (°C)
            RH: Relative humidity (%)
            LAI: Leaf area index (optional, uses current if None)
        
        Returns:
            Transpiration rate (L/hour)
        """
        if LAI is None:
            LAI = self.organs.LAI
        
        if LAI < 0.01:
            return 0.0
        
        # Calculate VPD (Vapor Pressure Deficit)
        # Saturation vapor pressure (kPa) - Tetens formula
        es = 0.611 * np.exp(17.27 * T / (T + 237.3))
        ea = es * (RH / 100.0)
        VPD = max(0.0, es - ea)  # kPa
        
        # Transpiration = f(LAI, VPD, water_stress)
        transp_base = self.params.transp_coef * LAI * VPD / 24.0  # L/hour
        
        # Water stress reduces transpiration
        water_stress = self._water_stress(self.soil_moisture)
        transp = transp_base * water_stress
        
        return transp
    
    def nutrient_uptake(self) -> float:
        """
        Calculate nutrient (N) uptake rate.
        
        Returns:
            N uptake (g/hour)
        """
        root_biomass = np.sum(self.organs.B_root)
        
        # Uptake proportional to root mass and solution concentration
        uptake_rate = (self.params.N_uptake_rate * root_biomass * 
                      self.solution_N / 24.0)  # g/hour
        
        return uptake_rate
    
    def biomass_allocation(self, available_biomass: float, development_stage: float) -> Dict[str, float]:
        """
        Allocate new biomass to organs based on development stage.
        
        Args:
            available_biomass: New biomass to allocate (g)
            development_stage: Plant age factor in [0, 1] (0=young, 1=mature)
        
        Returns:
            Dictionary with allocation to each organ type
        """
        # Interpolate partition coefficients based on development
        f_leaf = (self.params.partition_leaf_early * (1 - development_stage) + 
                 self.params.partition_leaf_late * development_stage)
        f_root = (self.params.partition_root_early * (1 - development_stage) + 
                 self.params.partition_root_late * development_stage)
        f_stem = (self.params.partition_stem_early * (1 - development_stage) + 
                 self.params.partition_stem_late * development_stage)
        
        # Normalize
        total = f_leaf + f_root + f_stem
        f_leaf /= total
        f_root /= total
        f_stem /= total
        
        return {
            'leaf': available_biomass * f_leaf,
            'stem': available_biomass * f_stem,
            'root': available_biomass * f_root
        }
    
    def step(self, light: float, temp: float, water_input: float = 0.0,
             nutrient_input: float = 0.0, RH: float = 60.0, 
             evaporation: float = 0.0) -> Dict[str, Any]:
        """
        Advance plant model by one timestep.
        
        Args:
            light: PAR or normalized light [0, 1] (if < 5, assumed normalized)
            temp: Temperature (°C)
            water_input: Water added (L)
            nutrient_input: Nutrients added (g or normalized [0,1])
            RH: Relative humidity (%)
            evaporation: Environmental evaporation (L)
        
        Returns:
            Dictionary with diagnostic information
        """
        # Convert normalized light to PAR if needed
        if light < 5.0:  # Assume normalized
            PAR = light * 1500.0  # FIXED: Changed from 1000 to 1500
        else:
            PAR = light
        
        
        # === FIX #1: UPDATE STRESS FACTORS FIRST ===
        self.stress_factors['water'] = self._water_stress(self.soil_moisture)
        self.stress_factors['temp'] = self._temperature_response(temp)
        self.stress_factors['nutrient'] = self._nutrient_stress(self.solution_N)
        # === CARBON BALANCE ===
        # Photosynthesis (gross)
        Pg = self.photosynthesis(PAR, temp)
        
        # Respiration
        Rm = self.respiration(temp)
        
        # Net assimilation (convert CO2 to CH2O biomass)
        # 44g CO2 -> 30g CH2O (dry biomass)
        net_assim = (Pg - Rm) * (30.0 / 44.0) * self.dt  # g dry biomass
        
        # Add to NSC pool
        self.organs.NSC += max(0.0, net_assim)
        
        # Growth respiration (30% of gross assimilation for synthesis)
        growth_resp = 0.3 * max(0.0, net_assim)
        net_growth = max(0.0, net_assim - growth_resp)
        
        # === BIOMASS ALLOCATION ===
        # Development stage (simple thermal time accumulation)
        if temp > self.params.T_min:
            self.organs.thermal_time += (temp - self.params.T_min) * self.dt
        
        development_stage = min(1.0, self.organs.thermal_time / 500.0)  # Arbitrary maturity at 500 degree-days
        
        # Allocate growth to organs
        if net_growth > 0.0:
            allocation = self.biomass_allocation(net_growth, development_stage)
            
            # Add to newest cohort (simplified - in full FSPM, new cohorts form periodically)
            self.organs.B_leaf[-1] += allocation['leaf']
            self.organs.B_stem[-1] += allocation['stem']
            self.organs.B_root[-1] += allocation['root']
        
        # === WATER BALANCE ===
        transp = self.transpiration(temp, RH) * self.dt  # L
        
        # Update soil moisture
        water_capacity = 5.0  # Liters (arbitrary container size)
        self.soil_moisture += (water_input - transp - evaporation) / water_capacity
        self.soil_moisture = np.clip(self.soil_moisture, 0.0, 1.0)
        
        # === NUTRIENT BALANCE ===
        N_uptake = self.nutrient_uptake() * self.dt  # g
        
        # Update solution nutrient
        if nutrient_input > 0:
            # If nutrient_input is normalized [0,1], scale it
            if nutrient_input < 1.0:
                nutrient_input *= 0.05  # Max 50mg N per application
        
        self.solution_N += (nutrient_input - N_uptake) / 1.0  # Normalize to pool size
        self.solution_N = np.clip(self.solution_N, 0.0, 1.0)
        
        self.organs.N_content += N_uptake
        
        # === UPDATE STRUCTURE ===
        self._update_lai()
        
        # Simple height model (allometric)
        total_biomass = np.sum(self.organs.B_leaf) + np.sum(self.organs.B_stem) + np.sum(self.organs.B_root)
        self.organs.plant_height = 0.05 + 0.3 * (total_biomass ** 0.4)  # meters
        
        # Update daily integrals for diagnostics
        self.daily_photosynthesis += Pg * self.dt
        self.daily_respiration += Rm * self.dt
        self.daily_transpiration += transp
        
        # === OUTPUT ===
        return {
            'photosynthesis_rate': Pg,
            'respiration_rate': Rm,
            'net_growth': net_growth,
            'transpiration_liters': transp,
            'LAI': self.organs.LAI,
            'biomass_total': total_biomass,
            'biomass_leaf': np.sum(self.organs.B_leaf),
            'biomass_stem': np.sum(self.organs.B_stem),
            'biomass_root': np.sum(self.organs.B_root),
            'NSC': self.organs.NSC,
            'N_content': self.organs.N_content,
            'soil_moisture': self.soil_moisture,
            'solution_N': self.solution_N,
            'stress_water': self.stress_factors['water'],
            'stress_temp': self.stress_factors['temp'],
            'stress_nutrient': self.stress_factors['nutrient'],
            'height_m': self.organs.plant_height,
            'development_stage': development_stage,
        }
    
    def get_state(self) -> Dict[str, float]:
        """Get current plant state for observations"""
        total_biomass = (np.sum(self.organs.B_leaf) + 
                        np.sum(self.organs.B_stem) + 
                        np.sum(self.organs.B_root))
        
        return {
            'canopy': np.clip(self.organs.LAI / 5.0, 0.0, 1.0),  # Normalize by max LAI
            'moisture': self.soil_moisture,
            'nutrient': self.solution_N,
            'biomass_fraction': np.clip(total_biomass / 50.0, 0.0, 1.0),  # Normalize
            'LAI': self.organs.LAI,
            'stress_water': self.stress_factors['water'],
            'stress_temp': self.stress_factors['temp'],
            'stress_nutrient': self.stress_factors['nutrient'],
        }
    
    def reset(self, initial_biomass: float = 1.0):
        """Reset plant to initial state"""
        self.organs = PlantOrgans()
        self.organs.B_leaf = np.array([initial_biomass * 0.5])
        self.organs.B_stem = np.array([initial_biomass * 0.2])
        self.organs.B_root = np.array([initial_biomass * 0.3])
        
        self.soil_moisture = 0.5
        self.solution_N = 0.5
        self.organs.NSC = 0.5
        self.organs.N_content = initial_biomass * 0.03
        
        self.daily_photosynthesis = 0.0
        self.daily_respiration = 0.0
        self.daily_transpiration = 0.0
        self.stress_factors = {'water': 1.0, 'temp': 1.0, 'nutrient': 1.0}
        
        self._update_lai()
    
    def is_dead(self, temp: float = None) -> bool:
        """Check if plant has died"""
        total_biomass = (np.sum(self.organs.B_leaf) + 
                        np.sum(self.organs.B_stem) + 
                        np.sum(self.organs.B_root))
        
        # Death conditions
        if total_biomass < 0.1:  # Below minimum viable biomass
            return True
        if self.soil_moisture < 0.05:  # Severe drought
            return True
        if temp is not None and (temp < 5.0 or temp > 40.0):  # Lethal temperatures
            return True
        
        return False