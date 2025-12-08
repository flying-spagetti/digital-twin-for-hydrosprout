# sim/soil_model_extended.py
"""
Extended SoilModel with macronutrients and micronutrients, pH-dependent availability,
uptake, dosing, leaching, mineralization and simple toxicity checks.

Pools are normalized 0..1 for simulation convenience. Units are abstracted; tune mapping
to real units when moving to real dosing.
"""

import math

MICRO_DEFAULTS = ["Fe", "Mn", "Zn", "Cu", "B", "Mo", "Se", "Cl"]

class SoilModelExtended:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        # macronutrient pools (normalized)
        self.soil_N = cfg.get("soil_N", 0.6)
        self.soil_P = cfg.get("soil_P", 0.45)
        self.soil_K = cfg.get("soil_K", 0.5)

        # micronutrient pools: dictionary keyed by chemical symbol
        base_micro = cfg.get("soil_micro", {})
        self.micro = {}
        for m in MICRO_DEFAULTS:
            self.micro[m] = float(base_micro.get(m, cfg.get(f"soil_{m.lower()}", 0.1)))

        # pH & environment
        self.pH = float(cfg.get("pH", 6.2))
        # leaching coefficients: macros more mobile than micros
        self.leach_coeff_macros = float(cfg.get("leach_coeff_macros", 0.05))
        self.leach_coeff_micros = float(cfg.get("leach_coeff_micros", 0.01))
        # slow mineralization rates
        self.mineralization_rate = float(cfg.get("mineralization", 0.0005))

        # toxicity thresholds (normalized) for quick checks; species may override
        default_tox = cfg.get("toxicity_thresholds", {})
        self.toxicity = {}
        for m in MICRO_DEFAULTS:
            self.toxicity[m] = float(default_tox.get(m, 1.0))  # 1.0 means unlikely by default

        # chelation factor: fraction of added chelated dose that becomes more available
        self.chelate_factor = float(cfg.get("chelate_factor", 0.6))

    def ec_proxy(self):
        """
        Simple EC/TDS proxy computed as weighted sum of pools (macros + micros).
        Scaled for monitoring and thresholding; not real EC units.
        """
        macro_sum = self.soil_N*1.2 + self.soil_P*0.9 + self.soil_K*1.0
        micro_sum = sum(self.micro[m] * 0.4 for m in self.micro)
        # scale up to 0..2.5 range
        return min(3.0, macro_sum + micro_sum)

    def ph_availability_factor(self, element):
        """
        Simple pH availability model:
        - Fe: low availability at high pH (>6.5)
        - Mn: reduces at high pH
        - Mo: more available at higher pH
        - others: mildly affected
        Returns factor 0..1 multiplier for uptake availability.
        """
        p = self.pH
        if element == "Fe":
            # availability falls off above 6.5
            return max(0.0, 1.0 - 0.5 * max(0.0, (p - 6.5)))
        if element == "Mn":
            return max(0.0, 1.0 - 0.4 * max(0.0, (p - 6.5)))
        if element == "Mo":
            return 0.5 + 0.1 * (p - 6.0)  # increases with pH
        # default mild effect
        return 1.0

    def add_dose(self, N=0.0, P=0.0, K=0.0, micro=None, chelated=False):
        """
        Add nutrient doses. micro is dict like {'Fe':0.01,'Zn':0.002}
        chelated: if True, apply improved availability fraction
        """
        self.soil_N = min(1.0, self.soil_N + float(N))
        self.soil_P = min(1.0, self.soil_P + float(P))
        self.soil_K = min(1.0, self.soil_K + float(K))
        micro = micro or {}
        for m, amt in micro.items():
            if m in self.micro:
                added = float(amt)
                if chelated:
                    added = added * (1.0 + self.chelate_factor)
                self.micro[m] = min(1.0, self.micro[m] + added)

    def mineralize(self):
        """Slowly increases available nutrients (organic -> mineral)"""
        self.soil_N = min(1.0, self.soil_N + self.mineralization_rate)
        self.soil_P = min(1.0, self.soil_P + self.mineralization_rate * 0.5)
        self.soil_K = min(1.0, self.soil_K + self.mineralization_rate * 0.6)
        for m in self.micro:
            self.micro[m] = min(1.0, self.micro[m] + self.mineralization_rate * 0.1)

    def uptake(self, demand):
        """
        demand: dict with keys 'N','P','K' (scalars or array-like per plant) and 'micro' dict of arrays or scalars
        Returns actual uptake dict: {'N':val,'P':val,'K':val,'micro':{...}} and reduces soil pools.
        Behavior:
          - Sum demand (if arrays) -> total requirement
          - Limit by available pool scaled by pH availability
          - Subtract actual uptake from soil pools
        """
        # handle macros
        def total(x):
            if hasattr(x, "__iter__"):
                return float(sum(x))
            return float(x)
        req_N = total(demand.get("N", 0.0))
        req_P = total(demand.get("P", 0.0))
        req_K = total(demand.get("K", 0.0))

        act_N = min(self.soil_N, req_N)
        act_P = min(self.soil_P, req_P)
        act_K = min(self.soil_K, req_K)

        # subtract proportionally
        if req_N > 0:
            self.soil_N = max(0.0, self.soil_N - act_N)
        if req_P > 0:
            self.soil_P = max(0.0, self.soil_P - act_P)
        if req_K > 0:
            self.soil_K = max(0.0, self.soil_K - act_K)

        # micros
        micro_req = demand.get("micro", {})
        micro_act = {}
        for m in self.micro:
            req = total(micro_req.get(m, 0.0))
            # availability reduced/increased by pH factor
            avail_factor = self.ph_availability_factor(m)
            avail = self.micro[m] * avail_factor
            actual = min(avail, req)
            # convert actual back into pool reduction in raw units (unscale by factor)
            if actual > 0:
                # when avail_factor <1, actual uptake will deplete the real pool proportionally
                reduce_amount = actual / max(1e-9, avail_factor)
                self.micro[m] = max(0.0, self.micro[m] - reduce_amount)
            micro_act[m] = actual

        return {
            "N": act_N,
            "P": act_P,
            "K": act_K,
            "micro": micro_act
        }

    def leach(self, water_liters, root_zone_vol_m3):
        """
        water_liters: liters draining through root zone this step.
        Remove fraction of pools proportional to water / root_zone volume scaled by coefficients.
        """
        if water_liters <= 0:
            return
        factor_macro = self.leach_coeff_macros * ((water_liters * 0.001) / max(1e-6, root_zone_vol_m3))
        factor_micro = self.leach_coeff_micros * ((water_liters * 0.001) / max(1e-6, root_zone_vol_m3))
        factor_macro = min(1.0, factor_macro)
        factor_micro = min(1.0, factor_micro)
        self.soil_N = max(0.0, self.soil_N * (1.0 - factor_macro))
        self.soil_P = max(0.0, self.soil_P * (1.0 - factor_macro))
        self.soil_K = max(0.0, self.soil_K * (1.0 - factor_macro))
        for m in self.micro:
            self.micro[m] = max(0.0, self.micro[m] * (1.0 - factor_micro))

    def toxicity_warnings(self, species_tox_thresholds=None):
        """
        Compare current pools to toxicity thresholds and return list of warnings.
        species_tox_thresholds: optional dict overrides for species-specific tolerance
        """
        warnings = []
        th = species_tox_thresholds or self.toxicity
        for m, pool in self.micro.items():
            thr = th.get(m, 1.0)
            if pool > thr:
                warnings.append((m, pool, thr))
        return warnings

    def status(self):
        s = {
            "soil_N": self.soil_N,
            "soil_P": self.soil_P,
            "soil_K": self.soil_K,
            "pH": self.pH,
            "ec": self.ec_proxy(),
        }
        for m in self.micro:
            s[f"soil_{m}"] = self.micro[m]
        return s
