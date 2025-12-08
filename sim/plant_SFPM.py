# sim/plant_structural.py
"""
Patched PlantStructural integrated with SoilModelExtended and species presets.

- Loads species presets from configs/species_presets_extended.yaml (fallback to basic presets)
- Uses SoilModelExtended for macro + micro nutrient pools, uptake, leaching, mineralization
- Computes per-plant nutrient demands and applies actual uptake to scale growth (nutrient limitation)
- Emits detailed diagnostics suitable for logging and the viz/analyze_sim_logs pipeline

Note: This file intentionally keeps normalized units for nutrient pools. When moving to
real dosing, supply conversion factors to map sim-units <-> mg/L or g per tray.

Also defines SAMPLE_REAL_IMAGE path (from session uploads) for quick visual composites.
"""

import os
import math
import numpy as np
import yaml
from pathlib import Path

# local imports (ensure sim/soil_model_extended.py exists)
try:
    from sim.soil_model import SoilModelExtended
except Exception:
    # If extended soil missing, fall back or raise
    raise

# Sample uploaded image path (from session uploads) - used by visualization utilities
SAMPLE_REAL_IMAGE = '/mnt/data/A_high-resolution_digital_photograph_showcases_an_.png'

# Default species presets file (extended)
SPECIES_PRESETS_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'species_presets_extended.yaml')

# -----------------------------
# Helper functions
# -----------------------------

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

# -----------------------------
# PlantStructural (patched)
# -----------------------------
class PlantStructural:
    def __init__(self, rows=8, cols=4, tray_area_m2=0.06, species='kale', soil_cfg=None, params=None, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.rows = rows
        self.cols = cols
        self.n = rows * cols
        self.tray_area = float(tray_area_m2)
        self.time = 0.0

        # Load base physiological params (defaults)
        base_p = {
            'T_opt': 20.0, 'T_sigma': 6.0,
            'k_maint': 0.0015, 'growth_resp_frac': 0.25,
            'SLA_cm2_per_g': 200.0,
            'percolation': 0.01,
            'root_zone_depth_m': 0.02,
        }
        if params:
            base_p.update(params)
        self.p = base_p

        # Species presets
        self.species_name = species
        self.species = self._load_species_preset(species)
        # override base A_max and SLA if present
        if 'A_max' in self.species:
            self.p['A_max'] = float(self.species['A_max'])
        if 'SLA_cm2_per_g' in self.species:
            self.p['SLA_cm2_per_g'] = float(self.species['SLA_cm2_per_g'])
        self.SLA_m2_g = self.p['SLA_cm2_per_g'] / 10000.0

        # Biomass pools per plant (g DW)
        self.B_leaf = np.full(self.n, 0.015)
        self.B_stem = np.full(self.n, 0.005)
        self.B_root = np.full(self.n, 0.01)
        self.LA = self.B_leaf * self.SLA_m2_g

        # Soil model
        self.soil = SoilModelExtended(cfg=soil_cfg or {})

        # Soil moisture (volumetric water content, 0..1)
        # Initialize soil_theta here so it's available before first step
        self.soil_theta = 0.35

        # Diagnostics placeholders
        self.LAI = self.compute_LAI()
        self.canopy_cover = self.LAI_to_cover(self.LAI)
        self.mold_prob = np.zeros(self.n)

    # -----------------------------
    # Species loader
    # -----------------------------
    def _load_species_preset(self, name):
        # Try extended presets first
        try:
            preset_path = Path(SPECIES_PRESETS_PATH).resolve()
            if preset_path.exists():
                with open(preset_path, 'r') as f:
                    data = yaml.safe_load(f)
                if name in data:
                    return data[name]
        except Exception:
            pass
        # fallback minimal defaults
        return {
            'A_max': 0.028,
            'SLA_cm2_per_g': 200.0,
            'alloc_leaf': 0.5,
            'alloc_stem': 0.3,
            'alloc_root': 0.2,
            'nutrient_demand': {'N': 0.5, 'P': 0.25, 'K': 0.4},
            'micronutrient_demand': {'Fe': 0.01, 'Mn': 0.004, 'Zn': 0.003, 'Cu': 0.0007},
        }

    # -----------------------------
    # Physiology
    # -----------------------------
    def light_response(self, I_norm):
        k = 0.2
        return I_norm / (I_norm + k)

    def temp_response(self, T):
        Topt = self.p.get('T_opt', 20.0)
        sigma = self.p.get('T_sigma', 6.0)
        return np.exp(-((T - Topt) ** 2) / (2 * sigma ** 2))

    def photosynthesis_per_plant(self, I_norm, T, nutrient_scale):
        Amax = float(self.p.get('A_max', self.species.get('A_max', 0.028)))
        light_fac = self.light_response(I_norm)
        temp_fac = self.temp_response(T)
        A = Amax * light_fac * temp_fac * nutrient_scale
        return np.full(self.n, A)

    # stomatal/transpiration same as earlier simplified
    def stomatal_conductance(self, vpd):
        gs_max = 0.4
        s = np.exp(-0.6 * vpd)
        return gs_max * s

    def transpiration_per_plant(self, T, RH, I_norm):
        es = 0.6108 * np.exp((17.27 * T) / (T + 237.3))
        ea = es * (RH / 100.0)
        vpd = np.maximum(0.0, es - ea)
        vpd_norm = clamp(vpd / 3.0, 0.0, 2.0)
        g_stom = self.stomatal_conductance(vpd_norm)
        LA = self.LA
        transp = 0.001 * g_stom * vpd_norm * (LA * 1000.0) * (0.5 + 0.5 * I_norm)
        transp = np.maximum(0.0, transp)
        return transp, vpd_norm

    # -----------------------------
    # Soil-water-nutrient helpers
    # -----------------------------
    def compute_LAI(self):
        return float(np.sum(self.LA) / max(1e-9, self.tray_area))

    def LAI_to_cover(self, LAI):
        k = 0.8
        return clamp(1.0 - math.exp(-k * LAI), 0.0, 1.0)

    def nutrient_demand_per_plant(self, growth_potential):
        """Return per-plant demand arrays for macros and micros based on species demands and growth potential."""
        nd = self.species.get('nutrient_demand', {})
        md = self.species.get('micronutrient_demand', {})
        # scale demand by growth potential (g DW) to get per-plant demand in normalized sim-units
        # growth_potential is array per plant (g DW)
        N_req = growth_potential * float(nd.get('N', 0.5))
        P_req = growth_potential * float(nd.get('P', 0.25))
        K_req = growth_potential * float(nd.get('K', 0.4))
        micro_req = {}
        for m, base in md.items():
            micro_req[m] = growth_potential * float(base)
        return {'N': N_req, 'P': P_req, 'K': K_req, 'micro': micro_req}

    # -----------------------------
    # Main step
    # -----------------------------
    def step(self, dt, env_state, water_liters=0.0, nutrient_dose=None):
        T = env_state.get('T', 22.0)
        RH = env_state.get('RH', 60.0)
        I_norm = env_state.get('I_norm', 0.5)

        # apply nutrient dose if provided (dict with macros and micro dict)
        if nutrient_dose:
            macros = {k: nutrient_dose.get(k, 0.0) for k in ('N', 'P', 'K')}
            micros = nutrient_dose.get('micro', None)
            chelated = nutrient_dose.get('chelated', False)
            self.soil.add_dose(N=macros['N'], P=macros['P'], K=macros['K'], micro=micros, chelated=chelated)

        # water addition -> update soil theta (simple)
        root_zone_vol = self.tray_area * self.p.get('root_zone_depth_m', 0.02)
        added_theta = (water_liters * 0.001) / max(1e-9, root_zone_vol)
        # store soil theta as derived metric (not in soil model extended but tracked here)
        if not hasattr(self, 'soil_theta'):
            self.soil_theta = 0.35
        self.soil_theta = clamp(self.soil_theta + added_theta, 0.0, 1.0)

        # nutrient factor provisional: combine macro availability into a single scalar
        # We'll compute demand and call soil.uptake to get actual uptake and derive per-plant nutrient_scale
        # 1) compute potential photosynthetic growth (without nutrient limit)
        # compute a temporary uniform nutrient_scale=1 for potential A
        potential_A = self.photosynthesis_per_plant(I_norm, T, nutrient_scale=1.0)
        growth_potential = (potential_A * dt) * (1.0 - self.p.get('growth_resp_frac', 0.25)) * 0.6

        # 2) compute per-plant nutrient demand arrays
        demand = self.nutrient_demand_per_plant(growth_potential)

        # prepare soil demand structure expected by SoilModelExtended (sum across plants internally)
        soil_demand = {
            'N': demand['N'],
            'P': demand['P'],
            'K': demand['K'],
            'micro': demand['micro']
        }
        # 3) call soil.uptake -> returns actual uptake totals (sums)
        uptake = self.soil.uptake(soil_demand)

        # 4) compute per-plant nutrient_scale: if actual < demand, scale down proportionally
        # convert totals back to per-plant scaling by ratio actual/required (use macros primarily)
        # avoid division by zero
        req_N_total = float(np.sum(demand['N']))
        actual_N = float(uptake.get('N', 0.0))
        macro_scale = 1.0
        if req_N_total > 0:
            macro_scale = actual_N / req_N_total
            macro_scale = clamp(macro_scale, 0.0, 1.0)
        # same for micronutrients: compute min across micro elements scale
        micro_scales = []
        for m, arr in demand['micro'].items():
            req_total = float(np.sum(arr))
            actual_m = float(uptake.get('micro', {}).get(m, 0.0))
            if req_total > 0:
                micro_scales.append(clamp(actual_m / req_total, 0.0, 1.0))
        micro_scale = min(micro_scales) if micro_scales else 1.0

        # final nutrient scale per plant = min(macro_scale, micro_scale)
        nutrient_scale = min(macro_scale, micro_scale)

        # 5) photosynthesis with nutrient limitation
        A_plants = self.photosynthesis_per_plant(I_norm, T, nutrient_scale)
        growth_potential = (A_plants * dt) * (1.0 - self.p.get('growth_resp_frac', 0.25)) * 0.6

        # maintenance respiration
        maint_resp = self.p.get('k_maint', 0.0015) * (self.B_leaf + self.B_stem + self.B_root) * dt
        growth_available = growth_potential - maint_resp
        growth_available = np.maximum(0.0, growth_available)

        # dynamic allocation influenced by nutrient status
        Nsoil = (self.soil.soil_N + 1e-9)
        alloc_root_adj = 0.5 * (1.0 - Nsoil)
        base_leaf = float(self.species.get('alloc_leaf', 0.5))
        base_root = float(self.species.get('alloc_root', 0.2)) + alloc_root_adj
        base_stem = float(self.species.get('alloc_stem', 0.3))
        tot = base_leaf + base_root + base_stem
        a_leaf = base_leaf / tot
        a_root = base_root / tot
        a_stem = base_stem / tot

        dB_leaf = growth_available * a_leaf
        dB_root = growth_available * a_root
        dB_stem = growth_available * a_stem

        self.B_leaf += dB_leaf
        self.B_root += dB_root
        self.B_stem += dB_stem

        # update LA
        self.LA = self.B_leaf * self.SLA_m2_g

        # transpiration & soil water update
        transp_per_plant, vpd_norm = self.transpiration_per_plant(T, RH, I_norm)
        total_transp = float(np.sum(transp_per_plant))
        evap = self.p.get('percolation', 0.01) * self.soil_theta * dt
        transp_theta = (total_transp * 0.001) / max(1e-9, root_zone_vol)
        # compute percolation/leaching: if soil_theta > field capacity, some water drains
        percolation_loss = self.p.get('percolation', 0.01) * self.soil_theta * dt
        # update soil theta
        self.soil_theta = clamp(self.soil_theta - transp_theta - evap - percolation_loss, 0.0, 1.0)

        # leach nutrients according to water percolation
        drained_liters = (percolation_loss * root_zone_vol) / 0.001  # reverse earlier conversion
        if drained_liters > 0:
            self.soil.leach(drained_liters, root_zone_vol)

        # mold probability update (per plant)
        mold_increase = (1.0 - vpd_norm) * (RH / 100.0) * 0.02 * dt
        mold_increase = mold_increase * (1.0 + 2.0 * clamp((self.soil_theta - 0.6), 0.0, 1.0))
        self.mold_prob = np.clip(self.mold_prob + mold_increase, 0.0, 1.0)

        # mineralize soil slowly
        self.soil.mineralize()

        # update LAI and cover
        self.LAI = self.compute_LAI()
        self.canopy_cover = self.LAI_to_cover(self.LAI)

        # toxicity warnings
        tox = self.soil.toxicity_warnings(self.species.get('micronutrient_toxicity_threshold', None))

        self.time += dt

        # diagnostics
        diagnostics = {
            'time': self.time,
            'A_plants_mean': float(np.mean(A_plants)),
            'growth_available_mean': float(np.mean(growth_available)),
            'B_leaf': self.B_leaf.copy(),
            'B_root': self.B_root.copy(),
            'B_stem': self.B_stem.copy(),
            'LA': self.LA.copy(),
            'LAI': float(self.LAI),
            'canopy_cover': float(self.canopy_cover),
            'soil_theta': float(self.soil_theta),
            'soil_status': self.soil.status(),
            'transp_total_liters': float(total_transp),
            'vpd_norm': float(vpd_norm),
            'mold_prob': self.mold_prob.copy(),
            'uptake': uptake,
            'tox_warnings': tox,
            'nutrient_scale': float(nutrient_scale),
        }
        return diagnostics

    def canopy_mask_params(self):
        coords = []
        radii = []
        for r in range(self.rows):
            for c in range(self.cols):
                x = (c + 0.5) / self.cols
                y = (r + 0.5) / self.rows
                coords.append((x, y))
                leaf_area = self.LA[r * self.cols + c]
                rnorm = clamp(np.sqrt(leaf_area / (0.01 + 1e-9)), 0.01, 0.5)
                radii.append(float(rnorm))
        return coords, np.array(radii)

# -----------------------------
# Demo script
# -----------------------------
if __name__ == '__main__':
    # Quick run to show behavior under different nutrient dosing
    sim = PlantStructural(rows=8, cols=4, tray_area_m2=0.06, species='kale', seed=42)
    hours = 168  # 7 days
    log = []
    for t in range(hours):
        # simple diurnal env
        hour = t % 24
        env = {'T': 24.0 if 7 <= hour <= 18 else 18.0, 'RH': 60.0 if 7 <= hour <= 18 else 75.0, 'I_norm': 0.8 if 7 <= hour <= 18 else 0.03}
        water = 0.25 if t % 12 == 0 else 0.0
        # dose small micronutrients on day 1
        dose = None
        if t == 0:
            dose = {'N': 0.02, 'P': 0.01, 'K': 0.015, 'micro': {'Fe': 0.002, 'Zn': 0.001}, 'chelated': True}
        d = sim.step(1.0, env, water_liters=water, nutrient_dose=dose)
        if t % 12 == 0:
            print(f"t={t}h LAI={d['LAI']:.3f} cover={d['canopy_cover']:.3f} soil_N={d['soil_status']['soil_N']:.3f} EC={d['soil_status']['ec']:.3f}")
        log.append(d)
    # write a small CSV
    try:
        import csv
        with open('structural_patched_demo.csv','w',newline='') as f:
            w=csv.writer(f)
            w.writerow(['time','LAI','canopy_cover','soil_N','soil_P','soil_K','ec'])
            for d in log:
                s=d['soil_status']
                w.writerow([d['time'],d['LAI'],d['canopy_cover'],s['soil_N'],s['soil_P'],s['soil_K'],s['ec']])
        print('Wrote structural_patched_demo.csv')
    except Exception:
        pass
