# sim/plant_structural.py
"""
Plant structural model for microgreens digital twin.

Features:
- Grid of individual plants (rows x cols)
- Biomass pools per plant: leaf / stem / root (g dry weight)
- Leaf Area (LA) per plant -> LAI and canopy fraction via Beer-Lambert
- Photosynthesis (light-limited saturating response) per plant with temperature modulation
- Respiration (maintenance) and growth respiration
- Allocation rules to leaf/stem/root (simple dynamic allocation)
- Soil water balance: transpiration (VPD-driven), evaporation, percolation, watering
- Nutrient uptake proportional to root biomass and soil nutrient concentration
- Mold probability computed from leaf wetness (high humidity or surface moisture) and stagnation

Dependencies: numpy
"""

import numpy as np

# -----------------------------
# Helper functions
# -----------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def clamp(x, a, b):
    return max(a, min(b, x))

# -----------------------------
# PlantStructural class
# -----------------------------
class PlantStructural:
    def __init__(self, rows=8, cols=4, tray_area_m2=0.06, params=None, seed=None):
        """
        rows, cols: grid layout of seedlings in a tray
        tray_area_m2: ground area of tray in m^2 (e.g., 0.06 m^2 ~ 300x200 mm)
        params: dictionary overriding default physiological parameters
        """
        if seed is not None:
            np.random.seed(seed)

        self.rows = rows
        self.cols = cols
        self.n = rows * cols
        self.tray_area = float(tray_area_m2)

        # default params
        p = {
            # photosynthesis
            "A_max": 0.025,     # max assimilation gC / (plant·hour) at saturating light (per-plant scale)
            "alpha": 0.8,       # initial slope of light response (dimensionless)
            "I_half": 0.2,      # half-saturation light (normalized 0..1)
            "T_opt": 20.0,      # optimal temperature °C
            "T_sigma": 6.0,     # temp response width

            # respiration
            "k_maint": 0.0015,  # maintenance respiration fraction per hour per g biomass
            "growth_resp_frac": 0.25,  # fraction of assimilates used for growth respiration

            # allocation
            "alloc_leaf": 0.5,  # baseline allocation to leaf fraction of growth
            "alloc_root": 0.2,
            "alloc_stem": 0.3,

            # leaf area specific
            "SLA": 200.0,       # specific leaf area cm^2 / gDW leaf (convert as needed)
            # Convert SLA to m^2 / g: SLA_m2_g = SLA / 10000
            # moisture transpiration
            "g_stom_max": 0.4,  # maximum stomatal conductance (mol m^-2 s^-1 scaled)
            "vpd_sensitivity": 0.6,
            "leaf_boundary_layer": 0.2,

            # soil water
            "soil_field_capacity": 0.5,
            "soil_wilt_point": 0.08,
            "root_zone_depth_m": 0.02,  # small for microgreens ~2 cm root zone
            "percolation": 0.01,  # per-hour percolation fraction

            # nutrient
            "nutrient_uptake_coeff": 0.02,  # per root g biomass per hour
            "nutrient_effect_k": 0.2,  # half-sat constant for nutrient effect

            # mold
            "mold_vpd_thresh": 0.6,
            "mold_humidity_effect": 1.2,
            "mold_growth_rate": 0.02
        }
        if params:
            p.update(params)
        self.p = p

        # initialize plant state arrays (per-plant)
        # biomass pools in g (dry weight)
        self.B_leaf = np.full(self.n, 0.015)  # 15 mg DW per seedling initial
        self.B_stem = np.full(self.n, 0.005)
        self.B_root = np.full(self.n, 0.01)
        # leaf area per plant (m^2) computed from B_leaf and SLA
        self.SLA_m2_g = p["SLA"] / 10000.0
        self.LA = self.B_leaf * self.SLA_m2_g  # m^2 per plant

        # soil state (single reservoir for tray)
        # volumetric water content (0..1 normalized)
        self.soil_theta = 0.35
        self.soil_nutrient = 0.6  # normalized nutrient concentration 0..1

        # canopy aggregated
        self.LAI = self.compute_LAI()  # m2 leaf area / m2 ground
        self.canopy_cover = self.LAI_to_cover(self.LAI)

        # plant health flags
        self.mold_prob = np.zeros(self.n)  # per plant mold probability

        # internal bookkeeping
        self.time = 0.0

    # -----------------------------
    # Core physiological functions
    # -----------------------------
    def light_response(self, I_norm):
        """
        Simplified saturating light response. I_norm is 0..1 (fraction of max PPFD)
        Returns relative photosynthetic rate factor (0..1)
        Uses a Michaelis-Menten-like curve
        """
        k = self.p["I_half"]
        return I_norm / (I_norm + k)

    def temp_response(self, T):
        """
        Gaussian temperature response centered on T_opt
        """
        Topt = self.p["T_opt"]
        sigma = self.p["T_sigma"]
        return np.exp(-((T - Topt) ** 2) / (2 * sigma ** 2))

    def photosynthesis_per_plant(self, I_norm, T, nutrient_factor):
        """
        Estimate carbon assimilation (g C per plant per hour) for each plant.
        Here we assume all plants see the same light fraction I_norm (top-down).
        nutrient_factor scales assimilation linearly (0..1)
        """
        # base rate scaled by light and temperature and nutrients
        Amax = self.p["A_max"]
        light_fac = self.light_response(I_norm)
        temp_fac = self.temp_response(T)
        A = Amax * light_fac * temp_fac * nutrient_factor
        # broadcast per-plant
        return np.full(self.n, A)

    def stomatal_conductance(self, vpd):
        """
        Very simple stomatal conductance model reducing with VPD.
        returns relative conductance (0..1)
        """
        gs_max = self.p["g_stom_max"]
        s = np.exp(-self.p["vpd_sensitivity"] * vpd)
        return gs_max * s

    def transpiration_per_plant(self, T, RH, I_norm):
        """
        Transpiration estimate (L water per plant per hour).
        We'll use a simplified approach:
        Transpiration ~ conductance * VPD * leaf_area
        Convert units loosely; values are scaled for microgreens.
        """
        # compute VPD approximation (kPa scaled to 0..1)
        # approximate saturation vapor pressure (Tetens), but we scale roughly
        es = 0.6108 * np.exp((17.27 * T) / (T + 237.3))  # kPa
        ea = es * (RH / 100.0)
        vpd = max(0.0, es - ea)  # kPa
        # normalize roughly to 0..1 by dividing by 3 kPa
        vpd_norm = clamp(vpd / 3.0, 0.0, 2.0)
        g_stom = self.stomatal_conductance(vpd_norm)  # relative
        # leaf area per plant
        LA = self.LA  # m2
        # transpiration factor (scaled): units liters per plant per hour
        transp = 0.001 * g_stom * vpd_norm * (LA * 1000.0) * (0.5 + 0.5 * I_norm)
        # ensure non-negative
        transp = np.maximum(0.0, transp)
        return transp, vpd_norm

    # -----------------------------
    # Soil water dynamics
    # -----------------------------
    def soil_evaporation(self, T, RH):
        """
        Bare-soil evaporation proxy (fraction of soil water lost per hour)
        increases with temperature and low humidity
        """
        evap_base = 0.005
        evap = evap_base * (1.0 + max(0.0, (T - 20.0) / 15.0)) * (1.0 + (1.0 - RH / 100.0))
        return clamp(evap, 0.0, 0.05)

    # -----------------------------
    # Conversion helpers
    # -----------------------------
    def compute_LAI(self):
        """Compute Leaf Area Index for full tray: total leaf area (m2) / ground area (m2)"""
        total_leaf_area = np.sum(self.LA)
        return total_leaf_area / self.tray_area

    def LAI_to_cover(self, LAI):
        """Compute canopy fractional cover (0..1) from LAI using Beer-Lambert (approx)."""
        # Beer-Lambert: Light attenuation ~ exp(-k*LAI). Define cover ~ 1 - exp(-k*LAI)
        k = 0.8
        cover = 1.0 - np.exp(-k * LAI)
        return clamp(cover, 0.0, 1.0)

    # -----------------------------
    # Uptake & allocation
    # -----------------------------
    def nutrient_effect(self):
        """Return nutrient scaling factor (0..1) from soil nutrient concentration."""
        N = self.soil_nutrient
        k = self.p["nutrient_effect_k"]
        return N / (N + k)

    def uptake_nutrients(self, dt):
        """Compute nutrient uptake by all roots (reduce soil nutrient)."""
        uptake_coeff = self.p["nutrient_uptake_coeff"]
        # uptake per plant proportional to root biomass
        uptake_per_plant = uptake_coeff * self.B_root * dt
        total_uptake = np.sum(uptake_per_plant)
        # scale soil nutrient down (simple mass-action)
        # soil_nutrient is normalized, we subtract fraction proportional to uptake
        self.soil_nutrient = clamp(self.soil_nutrient - (total_uptake * 0.01), 0.0, 1.0)
        return uptake_per_plant

    # -----------------------------
    # Main step function
    # -----------------------------
    def step(self, dt, env_state, water_liters=0.0, nutrient_dose=0.0):
        """
        Advance simulation by dt (hours).

        env_state: dict with keys:
            - 'T' : temperature (°C)
            - 'RH' : relative humidity (%)
            - 'I_norm' : incoming light normalized 0..1 (top-of-canopy)
        water_liters: liters added to the tray this step (distributed to soil)
        nutrient_dose: normalized nutrient added to soil (0..1)
        """

        T = env_state.get("T", 22.0)
        RH = env_state.get("RH", 60.0)
        I_norm = env_state.get("I_norm", 0.5)

        # 1) nutrient dose and water addition -> update soil reservoirs
        # Convert water liters to volumetric fraction: assume tray area * root_zone_depth gives soil volume
        root_zone_vol = self.tray_area * self.p["root_zone_depth_m"]  # m3
        # 1 liter = 0.001 m3
        added_theta = (water_liters * 0.001) / max(1e-6, root_zone_vol)
        self.soil_theta = clamp(self.soil_theta + added_theta, 0.0, 1.0)
        if nutrient_dose > 0.0:
            self.soil_nutrient = clamp(self.soil_nutrient + nutrient_dose, 0.0, 1.0)

        # 2) compute nutrient factor
        nutrient_factor = self.nutrient_effect()

        # 3) photosynthesis (assimilates per plant) -> g C per hour
        A_plants = self.photosynthesis_per_plant(I_norm, T, nutrient_factor)  # array shape n
        # translate carbon assimilates to biomass growth potential (g DW) after accounting respiration
        # convert C to DW roughly: assume 0.4 fraction carbon in DW (very rough). Use "growth_resp_frac"
        # growth potential per plant (g DW)
        growth_potential = (A_plants * dt) * (1.0 - self.p["growth_resp_frac"]) * 0.6  # scale factor to DW

        # 4) respiration
        maint_resp = self.p["k_maint"] * (self.B_leaf + self.B_stem + self.B_root) * dt  # per plant
        # maintenance respiration reduces overall available carbon; handle negative guard
        growth_available = growth_potential - maint_resp
        # any negative available -> set to zero (can't grow)
        growth_available = np.maximum(0.0, growth_available)

        # 5) nutrient uptake (reduces soil nutrient)
        uptake_per_plant = self.uptake_nutrients(dt)

        # 6) allocation to organs (simple proportional allocation, modulated by nutrient and water stress)
        # dynamic allocation: if nutrients low -> allocate more to root
        N = self.soil_nutrient
        alloc_root_adj = 0.5 * (1.0 - N)  # more root if nutrients low
        base_leaf = self.p["alloc_leaf"]
        base_root = self.p["alloc_root"] + alloc_root_adj
        base_stem = self.p["alloc_stem"]
        # normalize
        tot = base_leaf + base_root + base_stem
        a_leaf = base_leaf / tot
        a_root = base_root / tot
        a_stem = base_stem / tot

        dB_leaf = growth_available * a_leaf
        dB_root = growth_available * a_root
        dB_stem = growth_available * a_stem

        # 7) apply growth increments
        self.B_leaf += dB_leaf
        self.B_root += dB_root
        self.B_stem += dB_stem

        # 8) update leaf area (SLA conversion)
        self.LA = self.B_leaf * self.SLA_m2_g  # m2 per plant

        # 9) compute transpiration per plant and soil evaporation
        transp_per_plant, vpd_norm = self.transpiration_per_plant(T, RH, I_norm)
        total_transp = np.sum(transp_per_plant)  # liters per hour
        evap = self.soil_evaporation(T, RH) * dt
        # drain/transp reduces soil moisture
        # convert transpiration liters to volumetric fraction using root_zone_vol
        transp_vol = total_transp  # liters
        transp_theta = (transp_vol * 0.001) / max(1e-6, root_zone_vol)
        # percolation losses
        percolation_loss = self.p["percolation"] * self.soil_theta * dt
        self.soil_theta = clamp(self.soil_theta - transp_theta - evap - percolation_loss, 0.0, 1.0)

        # 10) mold probability: increases if RH high and leaf surface wetness (so low VPD) and low airflow
        # we track per-plant mold probability
        mold_increase = (1.0 - vpd_norm) * (RH / 100.0) * self.p["mold_growth_rate"] * dt
        # plants with very wet leaves (soil_theta high) get bigger mold increase
        mold_increase = mold_increase * (1.0 + 2.0 * clamp((self.soil_theta - 0.6), 0.0, 1.0))
        self.mold_prob = np.clip(self.mold_prob + mold_increase, 0.0, 1.0)

        # 11) update LAI, canopy cover
        self.LAI = self.compute_LAI()
        self.canopy_cover = self.LAI_to_cover(self.LAI)

        # 12) bookkeeping and return diagnostics
        self.time += dt

        diagnostics = {
            "time": self.time,
            "B_leaf": self.B_leaf.copy(),
            "B_root": self.B_root.copy(),
            "B_stem": self.B_stem.copy(),
            "LA": self.LA.copy(),
            "LAI": float(self.LAI),
            "canopy_cover": float(self.canopy_cover),
            "soil_theta": float(self.soil_theta),
            "soil_nutrient": float(self.soil_nutrient),
            "transp_total_liters": float(total_transp),
            "vpd_norm": float(vpd_norm),
            "mold_prob": self.mold_prob.copy(),
            "growth_added_leaf": dB_leaf.copy(),
            "growth_available": growth_available.copy()
        }
        return diagnostics

    # -----------------------------
    # Utility: produce a canopy mask parameterization
    # -----------------------------
    def canopy_mask_params(self):
        """
        Return a compact representation useful for synthetic image generation.

        Returns:
            - per_plant_centers: list of (x,y) coordinates in normalized tray coords (0..1)
            - per_plant_leaf_radius: array of radii (0..1) representing canopy blob size
        """
        # simple grid packing to coordinates
        coords = []
        radii = []
        for r in range(self.rows):
            for c in range(self.cols):
                x = (c + 0.5) / self.cols
                y = (r + 0.5) / self.rows
                coords.append((x, y))
                # radius proportional to sqrt(leaf area)
                leaf_area = self.LA[r * self.cols + c]
                # map leaf_area (m2) to 0..0.12 relative radius (tunable)
                rnorm = clamp(np.sqrt(leaf_area / (0.01 + 1e-9)), 0.01, 0.5)
                radii.append(float(rnorm))
        return coords, np.array(radii)

# -----------------------------
# Quick demo when run as script
# -----------------------------
if __name__ == "__main__":
    # small demo: 8x4 tray, step for 72 hours
    sim = PlantStructural(rows=8, cols=4, tray_area_m2=0.06, seed=42)
    hours = 72
    logs = []
    for t in range(hours):
        env = {"T": 25.0 if 6 <= (t % 24) <= 18 else 18.0, "RH": 60.0, "I_norm": 0.8 if 7 <= (t % 24) <= 17 else 0.05}
        # water lightly every 12 hours
        water = 0.0
        if t % 12 == 0:
            water = 0.25  # liters
        diag = sim.step(1.0, env, water_liters=water, nutrient_dose=0.0)
        if t % 6 == 0:
            print(f"t={t}h LAI={diag['LAI']:.3f} cover={diag['canopy_cover']:.3f} soil_theta={diag['soil_theta']:.3f}")
        logs.append(diag)
    # print final summary
    print("Final LAI:", sim.LAI, "Canopy cover:", sim.canopy_cover)
    centers, radii = sim.canopy_mask_params()
    print("Sample plant centers (first 4):", centers[:4])
    print("Sample radii (first 4):", radii[:4])
