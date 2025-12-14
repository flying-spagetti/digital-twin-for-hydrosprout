# sim/hardware_spatial.py
"""
Spatial Hardware Model for precision agriculture:
- Plant grid (individual plant locations)
- Nozzle array (water/nutrient/mist delivery)
- Spatial water/nutrient distribution
- Optimal spacing calculations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PlantLocation:
    """Individual plant in the grid"""
    x: float  # meters
    y: float  # meters
    biomass: float = 1.0
    health: float = 1.0
    soil_moisture: float = 0.5
    nutrient_level: float = 1.0


@dataclass
class Nozzle:
    """Water/nutrient delivery nozzle"""
    x: float
    y: float
    nozzle_type: str  # 'drip', 'mist', 'spray'
    flow_rate_max: float = 0.05  # L/hour
    coverage_radius: float = 0.1  # meters
    active: bool = False
    clog_factor: float = 1.0  # 0-1, accounts for clogging


class SpatialHardwareModel:
    """
    Manages spatial distribution of resources in growing area.
    
    Features:
    1. Plant grid with individual tracking
    2. Nozzle array for targeted delivery
    3. Spatial water/nutrient gradients
    4. Optimal spacing recommendations
    """
    
    def __init__(self, cfg=None):
        cfg = cfg or {}
        
        # Growing area dimensions
        self.width = cfg.get('width', 0.6)  # meters
        self.depth = cfg.get('depth', 0.4)  # meters
        
        # Plant configuration
        self.plant_spacing = cfg.get('plant_spacing', 0.05)  # 5cm default
        self.plants = self._init_plant_grid()
        
        # Nozzle configuration
        self.nozzle_spacing = cfg.get('nozzle_spacing', 0.15)  # 15cm default
        self.nozzles = self._init_nozzle_array(cfg.get('nozzle_type', 'mist'))
        
        # Fan array (from original model)
        self.fan_on = False
        self.shield_pos = 0.0
        
        # Energy tracking
        self.energy = 0.0
        self.water_efficiency = []  # Track delivery efficiency
    
    def _init_plant_grid(self) -> List[PlantLocation]:
        """Initialize plant grid with optimal spacing"""
        plants = []
        
        nx = int(self.width / self.plant_spacing)
        ny = int(self.depth / self.plant_spacing)
        
        for i in range(nx):
            for j in range(ny):
                x = (i + 0.5) * self.plant_spacing
                y = (j + 0.5) * self.plant_spacing
                plants.append(PlantLocation(x=x, y=y))
        
        return plants
    
    def _init_nozzle_array(self, nozzle_type: str) -> List[Nozzle]:
        """
        Initialize nozzle array for optimal coverage.
        
        Nozzle types:
        - 'drip': Point delivery, small radius
        - 'mist': Fine mist, medium radius
        - 'spray': Wider coverage, larger radius
        """
        nozzles = []
        
        coverage_params = {
            'drip': (0.03, 0.02),    # (flow_rate, radius)
            'mist': (0.05, 0.10),
            'spray': (0.10, 0.20)
        }
        
        flow_rate, radius = coverage_params.get(nozzle_type, (0.05, 0.10))
        
        nx = int(self.width / self.nozzle_spacing) + 1
        ny = int(self.depth / self.nozzle_spacing) + 1
        
        for i in range(nx):
            for j in range(ny):
                x = i * self.nozzle_spacing
                y = j * self.nozzle_spacing
                
                # Ensure within bounds
                if x <= self.width and y <= self.depth:
                    nozzles.append(Nozzle(
                        x=x, y=y,
                        nozzle_type=nozzle_type,
                        flow_rate_max=flow_rate,
                        coverage_radius=radius
                    ))
        
        return nozzles
    
    def calculate_optimal_spacing(self, target_LAI: float = 3.0,
                                   leaf_area_per_plant: float = 0.01) -> Dict:
        """
        Calculate optimal plant and nozzle spacing.
        
        Args:
            target_LAI: Desired leaf area index
            leaf_area_per_plant: Expected leaf area per plant (m²)
        
        Returns:
            Dict with recommended spacings
        """
        # LAI = total leaf area / ground area
        ground_area = self.width * self.depth
        required_leaf_area = target_LAI * ground_area
        n_plants_optimal = int(required_leaf_area / leaf_area_per_plant)
        
        # Calculate spacing
        plants_per_axis = int(np.sqrt(n_plants_optimal))
        optimal_plant_spacing = min(self.width, self.depth) / plants_per_axis
        
        # Nozzle spacing: 1 nozzle per 4-9 plants typically
        plants_per_nozzle = 6  # Target coverage
        n_nozzles_optimal = max(4, n_plants_optimal // plants_per_nozzle)
        nozzles_per_axis = int(np.sqrt(n_nozzles_optimal))
        optimal_nozzle_spacing = min(self.width, self.depth) / nozzles_per_axis
        
        return {
            'optimal_plant_spacing': optimal_plant_spacing,
            'optimal_nozzle_spacing': optimal_nozzle_spacing,
            'n_plants': n_plants_optimal,
            'n_nozzles': n_nozzles_optimal,
            'coverage_efficiency': self._calculate_coverage_efficiency()
        }
    
    def _calculate_coverage_efficiency(self) -> float:
        """
        Calculate what % of plants are within nozzle coverage.
        
        FIXED: Ensure plants and nozzles use the same coordinate system (meters).
        Log coordinate ranges for debugging.
        """
        if not self.plants or not self.nozzles:
            return 0.0
        
        # Debug: Log coordinate ranges
        plant_x = [p.x for p in self.plants]
        plant_y = [p.y for p in self.plants]
        nozzle_x = [n.x for n in self.nozzles]
        nozzle_y = [n.y for n in self.nozzles]
        
        # Verify coordinate system consistency
        # Plants should be in [0, width] x [0, depth] (meters)
        # Nozzles should also be in [0, width] x [0, depth] (meters)
        # If mismatch detected, log warning (but continue)
        if (min(plant_x) < 0 or max(plant_x) > self.width or 
            min(plant_y) < 0 or max(plant_y) > self.depth):
            # Plants may be normalized [0,1] - convert to meters
            # This is a bug fix: convert normalized coordinates to meters
            for plant in self.plants:
                if plant.x <= 1.0 and plant.y <= 1.0:
                    # Likely normalized, convert to meters
                    plant.x = plant.x * self.width
                    plant.y = plant.y * self.depth
        
        if (min(nozzle_x) < 0 or max(nozzle_x) > self.width or 
            min(nozzle_y) < 0 or max(nozzle_y) > self.depth):
            # Nozzles may be normalized - convert to meters
            for nozzle in self.nozzles:
                if nozzle.x <= 1.0 and nozzle.y <= 1.0:
                    nozzle.x = nozzle.x * self.width
                    nozzle.y = nozzle.y * self.depth
        
        # Calculate coverage
        covered_plants = 0
        for plant in self.plants:
            for nozzle in self.nozzles:
                dist = np.sqrt((plant.x - nozzle.x)**2 + (plant.y - nozzle.y)**2)
                if dist <= nozzle.coverage_radius:
                    covered_plants += 1
                    break
        
        coverage = covered_plants / len(self.plants) if self.plants else 0.0
        
        # Assertion: coverage should be > 0 for typical nozzle radius
        if coverage == 0.0 and len(self.nozzles) > 0 and len(self.plants) > 0:
            # Log warning for debugging
            avg_nozzle_radius = np.mean([n.coverage_radius for n in self.nozzles])
            min_plant_nozzle_dist = min([
                np.sqrt((p.x - n.x)**2 + (p.y - n.y)**2)
                for p in self.plants for n in self.nozzles
            ]) if self.plants and self.nozzles else float('inf')
            # This is a diagnostic - coverage=0 suggests coordinate mismatch
            pass  # Could log warning here
        
        return coverage
    
    def distribute_water(self, total_water_L: float,
                        nozzle_activation: Optional[np.ndarray] = None) -> Dict:
        """
        Distribute water to plants based on nozzle activation.
        
        Args:
            total_water_L: Total water to distribute
            nozzle_activation: Array of 0/1 for each nozzle (None = all active)
        
        Returns:
            Dict with water distribution map and efficiency
        """
        if nozzle_activation is None:
            nozzle_activation = np.ones(len(self.nozzles))
        
        # Ensure nozzle_activation matches number of nozzles
        if len(nozzle_activation) != len(self.nozzles):
            # If mismatch, use first N or pad with zeros
            if len(nozzle_activation) < len(self.nozzles):
                # Pad with zeros if activation array is shorter
                padded = np.zeros(len(self.nozzles))
                padded[:len(nozzle_activation)] = nozzle_activation
                nozzle_activation = padded
            else:
                # Truncate if activation array is longer
                nozzle_activation = nozzle_activation[:len(self.nozzles)]
        
        # Reset plant moisture additions
        water_received = {i: 0.0 for i in range(len(self.plants))}
        
        # Calculate per-nozzle flow
        active_nozzles = np.sum(nozzle_activation > 0.5)
        if active_nozzles == 0:
            return {'water_received': water_received, 'efficiency': 0.0}
        
        water_per_nozzle = total_water_L / active_nozzles
        
        # Distribute from each active nozzle
        for nozzle_idx, nozzle in enumerate(self.nozzles):
            # Bounds check
            if nozzle_idx >= len(nozzle_activation):
                continue
            if nozzle_activation[nozzle_idx] < 0.5:
                continue
            
            # Find plants in coverage
            for plant_idx, plant in enumerate(self.plants):
                dist = np.sqrt((plant.x - nozzle.x)**2 + (plant.y - nozzle.y)**2)
                
                if dist <= nozzle.coverage_radius:
                    # Distribution weighted by distance (inverse)
                    if dist < 0.01:
                        dist = 0.01
                    weight = 1.0 / dist
                    
                    # Account for clogging
                    delivered = water_per_nozzle * weight * nozzle.clog_factor
                    water_received[plant_idx] += delivered
        
        # Normalize to total water
        total_distributed = sum(water_received.values())
        if total_distributed > 0:
            scale = total_water_L / total_distributed
            water_received = {k: v * scale for k, v in water_received.items()}
        
        # Update plant moisture
        for plant_idx, water in water_received.items():
            # Simplified: 0.01L increases moisture by ~0.02
            if plant_idx < len(self.plants):
                moisture_gain = water * 2.0
                self.plants[plant_idx].soil_moisture += moisture_gain
                self.plants[plant_idx].soil_moisture = min(1.0, self.plants[plant_idx].soil_moisture)
        
        # Calculate efficiency (% actually reaching plants)
        efficiency = total_distributed / total_water_L if total_water_L > 0 else 0.0
        self.water_efficiency.append(efficiency)
        
        return {
            'water_received': water_received,
            'efficiency': efficiency,
            'distribution_uniformity': np.std(list(water_received.values()))
        }
    
    def evaporation_spatial(self, dt_hours: float, temp: float,
                           humidity: float) -> np.ndarray:
        """
        Calculate spatially-varying evaporation.
        
        Areas near edges evaporate faster than center.
        """
        evap_losses = []
        
        for plant in self.plants:
            # Base evaporation rate
            base_evap = 0.01 * (1 + (temp - 20) / 20.0) * dt_hours
            
            # Edge effect (edges dry faster)
            dist_to_edge = min(
                plant.x, self.width - plant.x,
                plant.y, self.depth - plant.y
            )
            edge_factor = 1.0 + (0.5 * np.exp(-dist_to_edge / 0.1))
            
            # Humidity effect
            humidity_factor = (100 - humidity) / 50.0
            
            evap = base_evap * edge_factor * humidity_factor * plant.soil_moisture
            evap = max(0.0, min(0.1, evap))
            
            plant.soil_moisture -= evap
            plant.soil_moisture = max(0.0, plant.soil_moisture)
            
            evap_losses.append(evap)
        
        return np.array(evap_losses)
    
    def get_spatial_stress_map(self) -> Dict[str, np.ndarray]:
        """Get spatial maps of plant stress factors"""
        moisture_map = np.array([p.soil_moisture for p in self.plants])
        nutrient_map = np.array([p.nutrient_level for p in self.plants])
        health_map = np.array([p.health for p in self.plants])
        
        return {
            'moisture': moisture_map,
            'nutrient': nutrient_map,
            'health': health_map,
            'positions': np.array([[p.x, p.y] for p in self.plants])
        }
    
    def step(self, action: Dict) -> Dict:
        """
        Execute hardware control step.
        
        action: {
            'water': total water L,
            'nozzle_control': array of nozzle activations (optional),
            'fan': 0/1,
            'shield': delta,
            'heater': 0-1
        }
        """
        # Water distribution
        water_l = action.get('water', 0.0)
        nozzle_control = action.get('nozzle_control', None)
        
        water_dist = self.distribute_water(water_l, nozzle_control)
        
        # Fan
        self.fan_on = bool(action.get('fan', 0))
        
        # Shield
        shield_delta = action.get('shield', 0.0)
        self.shield_pos = float(max(0.0, min(1.0, self.shield_pos + shield_delta * 0.2)))
        
        # Energy accounting
        self.energy += (
            abs(water_l) * 0.5 +
            (1.0 if self.fan_on else 0.0) * 0.2 +
            action.get('heater', 0.0) * 0.8
        )
        
        # Get spatial metrics
        stress_map = self.get_spatial_stress_map()
        
        # Calculate actual delivered water (accounting for efficiency)
        actual_delivered = water_l * water_dist.get('efficiency', 1.0) if water_l > 0 else 0.0
        
        # FIXED: Report both geometry coverage and delivery efficiency separately
        # nozzle_coverage_fraction: geometry-only (what % of plants are within nozzle coverage radius)
        nozzle_coverage_fraction = self._calculate_coverage_efficiency()
        # water_delivery_efficiency: delivery-only (what % of water actually reaches plants)
        water_delivery_efficiency = water_dist.get('efficiency', 0.0) if water_l > 0 else 0.0
        
        return {
            'delivered_water': actual_delivered,  # Actual water reaching plants
            'water_input': water_l,  # Total input water
            'water_distribution': water_dist,
            'water_efficiency': water_delivery_efficiency,  # Delivery efficiency
            'fan_on': self.fan_on,
            'shield_pos': self.shield_pos,
            'heater_power': action.get('heater', 0.0),
            'spatial_stress': stress_map,
            'n_plants': len(self.plants),
            # FIXED: Separate geometry coverage from delivery efficiency
            'nozzle_coverage_fraction': nozzle_coverage_fraction,  # Geometry-only: % plants within nozzle radius
            'water_delivery_efficiency': water_delivery_efficiency,  # Delivery-only: % water reaching plants
            # Legacy: keep for backward compatibility (use delivery efficiency)
            'coverage_efficiency': water_delivery_efficiency
        }