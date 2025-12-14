# sim/nutrient_model.py
"""
Comprehensive Nutrient Solution Model for hydroponics:
- NPK (Nitrogen, Phosphorus, Potassium) tracking
- Micronutrients (Ca, Mg, Fe, etc.)
- EC (Electrical Conductivity)
- pH dynamics
- Solution temperature
- Nutrient uptake by plants
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class NutrientRecipe:
    """Standard nutrient concentrations (ppm = mg/L)"""
    N: float = 150.0      # Nitrogen
    P: float = 50.0       # Phosphorus
    K: float = 200.0      # Potassium
    Ca: float = 160.0     # Calcium
    Mg: float = 50.0      # Magnesium
    S: float = 65.0       # Sulfur
    Fe: float = 3.0       # Iron
    Mn: float = 0.5       # Manganese
    Zn: float = 0.3       # Zinc
    Cu: float = 0.1       # Copper
    B: float = 0.5        # Boron
    Mo: float = 0.05      # Molybdenum


class NutrientSolutionModel:
    """
    Models nutrient solution dynamics in a hydroponic reservoir.
    
    Key processes:
    1. Plant uptake (preferential uptake of different nutrients)
    2. Precipitation (pH-dependent nutrient availability)
    3. EC drift
    4. pH buffering and drift
    5. Temperature effects on uptake rates
    """
    
    def __init__(self, cfg=None):
        cfg = cfg or {}
        
        # Reservoir properties
        self.volume_L = cfg.get('reservoir_volume_L', 10.0)
        self.temperature = cfg.get('initial_temp', 20.0)  # °C
        
        # Initialize nutrients from recipe
        recipe = cfg.get('recipe', NutrientRecipe())
        self.N = recipe.N * self.volume_L / 1000.0  # Convert ppm to grams
        self.P = recipe.P * self.volume_L / 1000.0
        self.K = recipe.K * self.volume_L / 1000.0
        self.Ca = recipe.Ca * self.volume_L / 1000.0
        self.Mg = recipe.Mg * self.volume_L / 1000.0
        self.S = recipe.S * self.volume_L / 1000.0
        
        # Micronutrients (ppm)
        self.Fe = recipe.Fe
        self.Mn = recipe.Mn
        self.Zn = recipe.Zn
        self.Cu = recipe.Cu
        self.B = recipe.B
        self.Mo = recipe.Mo
        
        # Solution chemistry
        self.pH = cfg.get('initial_pH', 6.0)
        self.EC = self._calculate_EC()  # mS/cm
        
        # Target ranges for optimal growth
        self.target_pH_min = 5.5
        self.target_pH_max = 6.5
        self.target_EC_min = 1.2
        self.target_EC_max = 2.4
        
        # Uptake ratios (relative to N uptake)
        self.uptake_ratios = {
            'N': 1.0,
            'P': 0.15,
            'K': 0.8,
            'Ca': 0.3,
            'Mg': 0.1,
            'S': 0.2,
        }
        
        # pH buffer capacity (affects pH drift rate)
        self.buffer_capacity = 0.5
    
    def _calculate_EC(self) -> float:
        """
        Calculate electrical conductivity from ion concentrations.
        
        Simplified EC = f(total dissolved salts)
        EC in mS/cm
        """
        # Convert back to ppm
        N_ppm = (self.N / self.volume_L) * 1000.0
        P_ppm = (self.P / self.volume_L) * 1000.0
        K_ppm = (self.K / self.volume_L) * 1000.0
        Ca_ppm = (self.Ca / self.volume_L) * 1000.0
        Mg_ppm = (self.Mg / self.volume_L) * 1000.0
        S_ppm = (self.S / self.volume_L) * 1000.0
        
        # EC estimation (empirical formula)
        # Roughly: 1 mS/cm ≈ 640-700 ppm TDS
        TDS = N_ppm + P_ppm + K_ppm + Ca_ppm + Mg_ppm + S_ppm
        EC = TDS / 650.0
        
        return max(0.1, EC)
    
    def get_nutrient_availability(self) -> Dict[str, float]:
        """
        Get nutrient availability factors (0-1) based on pH.
        
        Different nutrients have optimal pH ranges:
        - N: 5.5-8.0 (broad)
        - P: 5.5-6.5 (narrow)
        - K: 5.5-7.5
        - Ca: 5.5-7.0
        - Mg: 6.0-8.0
        - Fe: 5.0-6.5 (critical)
        """
        pH = self.pH
        
        def availability_curve(pH: float, optimal: float, width: float) -> float:
            """Gaussian-like availability"""
            return np.exp(-((pH - optimal) ** 2) / (2 * width ** 2))
        
        return {
            'N': availability_curve(pH, 6.5, 1.2),
            'P': availability_curve(pH, 6.0, 0.5),
            'K': availability_curve(pH, 6.5, 1.0),
            'Ca': availability_curve(pH, 6.5, 0.8),
            'Mg': availability_curve(pH, 7.0, 1.0),
            'S': availability_curve(pH, 6.5, 1.0),  # Sulfur availability
            'Fe': availability_curve(pH, 5.8, 0.7),
        }
    
    def plant_uptake(self, root_biomass: float, dt_hours: float,
                     temp_effect: float = 1.0) -> Dict[str, float]:
        """
        Simulate nutrient uptake by plants.
        
        Args:
            root_biomass: Root dry weight (g)
            dt_hours: Time step
            temp_effect: Temperature factor (0-1)
        
        Returns:
            Dict of nutrient masses consumed (g)
        """
        dt_days = dt_hours / 24.0
        
        # Base uptake rate (function of root biomass)
        base_uptake_N = 0.02 * root_biomass * dt_days * temp_effect
        
        # Get availability modifiers
        availability = self.get_nutrient_availability()
        
        # Calculate actual uptake for each nutrient
        uptake = {}
        
        # N uptake (limited by concentration and availability)
        # FIXED: Use sigmoid instead of hard clip to preserve gradients
        N_ppm = (self.N / self.volume_L) * 1000.0
        # Sigmoid saturation: smooth transition, preserves gradients
        # Use tanh-based sigmoid: 0.5 * (tanh((N_ppm - 50) / 20) + 1)
        # This gives smooth saturation around 50-100 ppm
        N_factor = 0.5 * (np.tanh((N_ppm - 50.0) / 20.0) + 1.0)
        N_factor = np.clip(N_factor, 0.0, 1.0)  # Safety clip
        uptake['N'] = base_uptake_N * N_factor * availability['N']
        
        # Other nutrients follow ratios
        for nutrient, ratio in self.uptake_ratios.items():
            if nutrient == 'N':
                continue
            
            ppm = (getattr(self, nutrient) / self.volume_L) * 1000.0
            conc_factor = min(1.0, ppm / 50.0)  # Adjust threshold per nutrient
            # Get availability, default to 0.8 if not in availability dict
            nutrient_availability = availability.get(nutrient, 0.8)
            uptake[nutrient] = uptake['N'] * ratio * conc_factor * nutrient_availability
        
        # Deplete reservoir
        self.N -= uptake['N']
        self.P -= uptake['P']
        self.K -= uptake['K']
        self.Ca -= uptake['Ca']
        self.Mg -= uptake['Mg']
        self.S -= uptake['S']
        
        # Prevent negative concentrations
        self.N = max(0.0, self.N)
        self.P = max(0.0, self.P)
        self.K = max(0.0, self.K)
        self.Ca = max(0.0, self.Ca)
        self.Mg = max(0.0, self.Mg)
        self.S = max(0.0, self.S)
        
        return uptake
    
    def add_nutrients(self, nutrient_doses: Dict[str, float]):
        """
        Add nutrients to reservoir (e.g., fertigation event).
        
        Args:
            nutrient_doses: Dict of {nutrient_name: mass_in_grams}
        """
        for nutrient, amount in nutrient_doses.items():
            if hasattr(self, nutrient):
                current = getattr(self, nutrient)
                setattr(self, nutrient, current + amount)
    
    def pH_drift(self, dt_hours: float, plant_uptake_N: float):
        """
        Model pH drift due to preferential ion uptake.
        
        Typically:
        - NO3- uptake raises pH (plants export OH-)
        - NH4+ uptake lowers pH (plants export H+)
        
        Simplified: N uptake causes slight pH rise
        """
        dt_days = dt_hours / 24.0
        
        # pH tends to rise with nitrate uptake
        pH_change = (plant_uptake_N * 0.1 / self.buffer_capacity) * dt_days
        
        # Natural drift toward neutral
        drift_to_neutral = (7.0 - self.pH) * 0.01 * dt_days
        
        self.pH += pH_change + drift_to_neutral
        self.pH = np.clip(self.pH, 4.0, 8.0)
    
    def adjust_pH(self, target_pH: float, adjustment_strength: float = 0.5):
        """
        Adjust pH toward target (simulates pH up/down dosing).
        
        Args:
            target_pH: Desired pH
            adjustment_strength: Rate of adjustment (0-1)
        """
        delta = (target_pH - self.pH) * adjustment_strength
        self.pH += delta
        self.pH = np.clip(self.pH, 4.0, 8.0)
    
    def water_consumption(self, water_used_L: float):
        """
        Account for water consumed (concentrates nutrients).
        
        Args:
            water_used_L: Water lost to evapotranspiration
        """
        if water_used_L > 0.1 and self.volume_L > 1.0:
            self.volume_L -= water_used_L
            # Nutrients remain, concentration increases
            # (Already tracked as grams, ppm will rise)
    
    def water_addition(self, water_added_L: float):
        """Add fresh water (dilutes nutrients)."""
        self.volume_L += water_added_L
    
    def step(self, root_biomass: float, dt_hours: float, 
             water_used_L: float = 0.0, temp: float = 20.0,
             nutrient_dose: Optional[Dict[str, float]] = None,
             pH_control: bool = False) -> Dict:
        """
        Simulate one time step of nutrient dynamics.
        
        Returns:
            State dict with all nutrient concentrations and metrics
        """
        self.temperature = temp
        
        # Temperature effect on uptake (Q10 rule)
        temp_effect = 2.0 ** ((temp - 20.0) / 10.0)
        temp_effect = np.clip(temp_effect, 0.5, 2.0)
        
        # Plant uptake
        uptake = self.plant_uptake(root_biomass, dt_hours, temp_effect)
        
        # pH drift
        self.pH_drift(dt_hours, uptake['N'])
        
        # pH control (automatic) - but only if agent is not actively controlling
        # FIXED: Disable auto pH control when agent provides pH_adjust action
        # This is handled in gym_env.py step() - if pH_adjust > 0.1, auto-control is disabled
        if pH_control:
            target = 6.0
            if self.pH < self.target_pH_min:
                self.adjust_pH(target, 0.3)
            elif self.pH > self.target_pH_max:
                self.adjust_pH(target, 0.3)
        
        # Water dynamics
        self.water_consumption(water_used_L)
        
        # Add nutrients if dosed
        if nutrient_dose:
            self.add_nutrients(nutrient_dose)
        
        # Recalculate EC
        self.EC = self._calculate_EC()
        
        # Get availability
        availability = self.get_nutrient_availability()
        
        # Return state
        return {
            'N_ppm': (self.N / self.volume_L) * 1000.0,
            'P_ppm': (self.P / self.volume_L) * 1000.0,
            'K_ppm': (self.K / self.volume_L) * 1000.0,
            'Ca_ppm': (self.Ca / self.volume_L) * 1000.0,
            'Mg_ppm': (self.Mg / self.volume_L) * 1000.0,
            'EC': self.EC,
            'pH': self.pH,
            'temperature': self.temperature,
            'volume_L': self.volume_L,
            'availability': availability,
            'uptake': uptake,
            'N_stress': min(1.0, ((self.N / self.volume_L) * 1000.0) / 100.0),
            'EC_stress': 1.0 - abs(self.EC - 1.8) / 1.8,  # Optimal around 1.8
            'pH_stress': availability['N']  # Use N as proxy
        }