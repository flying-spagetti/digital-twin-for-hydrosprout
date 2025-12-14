# Observation and Action Schema Documentation

This document describes the authoritative observation and action schemas used in the Digital Twin RL environment.

## Observation Schema

The observation vector is constructed from multiple subsystems. The exact order is defined by `OBS_KEYS` in `rl/gym_env.py`.

### Structure

Total dimensions: **31 + n_peltiers** (where n_peltiers is typically 4, so 35 total)

#### Plant State (11 features)
1. `plant_biomass_fraction` - Normalized biomass fraction [0, 1]
2. `plant_moisture` - Soil moisture [0, 1]
3. `plant_nutrient` - Nutrient availability [0, 1]
4. `plant_LAI` - Leaf Area Index, normalized by 5.0 [0, 1]
5. `plant_stress_water` - Water stress factor [0, 1]
6. `plant_stress_temp` - Temperature stress factor [0, 1]
7. `plant_stress_nutrient` - Nutrient stress factor [0, 1]
8. `plant_height` - Plant height, normalized by 0.5 [0, 1]
9. `plant_NSC` - Non-structural carbohydrates, normalized by 5.0 [0, 1]
10. `plant_N_content` - Nitrogen content [0, 1]
11. `plant_total_biomass` - Total biomass, normalized by 50.0 [0, 1]

#### Environment State (8 features)
12. `env_T_top` - Top zone temperature, normalized: (T-25)/10 [-1, 1]
13. `env_T_middle` - Middle zone temperature, normalized: (T-25)/10 [-1, 1]
14. `env_T_bottom` - Bottom zone temperature, normalized: (T-25)/10 [-1, 1]
15. `env_temp_stress` - Temperature stress factor [0, 1]
16. `env_RH_top` - Top zone relative humidity [0, 1] (0-100%)
17. `env_RH_middle` - Middle zone relative humidity [0, 1] (0-100%)
18. `env_RH_bottom` - Bottom zone relative humidity [0, 1] (0-100%)
19. `env_CO2` - CO2 concentration, normalized by 2000.0 [0, 1] (0-2000 ppm)

#### Nutrient State (5 features)
20. `nutrient_EC` - Electrical conductivity, normalized by 3.0 [0, 1] (0-3 mS/cm)
21. `nutrient_pH` - pH, normalized: (pH-4)/4 [0, 1] (4-8)
22. `nutrient_N_ppm` - Nitrogen concentration, normalized by 100.0 [0, 1] (0-100 ppm)
23. `nutrient_EC_stress` - EC stress factor [0, 1]
24. `nutrient_pH_stress` - pH stress factor [0, 1]

#### Hardware State (5 + n_peltiers features)
25. `hw_shield_pos` - Light shield position [0, 1]
26. `hw_fan_on` - Fan state [0, 1] (binary)
27. `hw_moisture_std` - Spatial moisture standard deviation [0, 1]
28. `hw_coverage_efficiency` - Nozzle coverage efficiency [0, 1]
29. `hw_water_efficiency` - Water delivery efficiency [0, 1]
30-33. `hw_peltier_0` through `hw_peltier_3` - Peltier module states [0, 1] (normalized from [-1, 1] power range)

#### Time Encoding (2 features)
34. `time_hour_sin` - Hour of day (sine encoding) [0, 1]
35. `time_hour_cos` - Hour of day (cosine encoding) [0, 1]

### Decoding Observations

**Never hardcode indices!** Always use `OBS_KEYS`:

```python
# Correct way:
obs_keys = env.OBS_KEYS
obs_dict = {key: obs[i] for i, key in enumerate(obs_keys)}
temp = obs_dict['env_T_middle'] * 10.0 + 25.0  # Denormalize

# Wrong way:
temp = obs[13] * 10.0 + 25.0  # DON'T DO THIS
```

The environment provides `info['obs_keys']` and `info['observed_state']` for debugging.

## Action Schema

Actions are provided as a Dict space with the following keys:

### Action Keys and Scaling

1. **`water_total`** [0, 1]
   - Scale: 0.05 L per unit
   - Applied: `water_L = water_total * 0.05`
   - Purpose: Water delivery to plants

2. **`fan`** [0, 1] (Discrete: 0=off, 1=on)
   - Applied: `fan_on = bool(fan > 0.5)`
   - Purpose: Air circulation

3. **`shield_delta`** [-1, 1]
   - Scale: 0.2 per unit
   - Applied: `shield_pos += shield_delta * 0.2`
   - Purpose: Light shield adjustment

4. **`heater`** [0, 1]
   - Scale: 200 W per unit
   - Applied: `heater_power = heater * 200.0`
   - Purpose: Heating

5. **`peltier_controls`** [-1, 1] (array of length n_peltiers)
   - Applied: Direct power control (-1=cooling, +1=heating)
   - Purpose: Peltier cooling/heating modules

6. **`dose_N`** [0, 1]
   - Scale: 0.5 g per unit
   - Applied: `N_dose = dose_N * 0.5`
   - Purpose: Nitrogen dosing

7. **`dose_P`** [0, 1]
   - Scale: 0.1 g per unit
   - Applied: `P_dose = dose_P * 0.1`
   - Purpose: Phosphorus dosing

8. **`dose_K`** [0, 1]
   - Scale: 0.3 g per unit
   - Applied: `K_dose = dose_K * 0.3`
   - Purpose: Potassium dosing

9. **`pH_adjust`** [-1, 1]
   - Applied: `target_pH = 6.0 + pH_adjust * 1.0`
   - Purpose: pH adjustment (disables auto pH control if |pH_adjust| > 0.1)

10. **`nozzle_mask`** [0, 1] (MultiBinary array of length n_nozzles)
    - Applied: Binary mask for nozzle activation
    - Purpose: Spatial water distribution control

11. **`co2_inject`** [0, 1]
    - Scale: 10.0 L/hour per unit
    - Applied: `co2_L_per_hour = co2_inject * 10.0`
    - Purpose: CO2 enrichment

### Action Logging

The environment logs both:
- **`raw_action`**: Policy output (before scaling/clipping)
- **`applied_action`**: Actual action applied to environment (after scaling/clipping)

**Always use `applied_action` for diagnostics and plots!**

## Debug Information

The `info` dict returned by `env.step()` contains:

```python
info = {
    'true_state': {...},        # Physical state in real units
    'observed_state': {...},    # Decoded observation (what agent sees)
    'obs_keys': [...],          # Observation schema
    'action_keys': {...},        # Action schema with scaling factors
    'raw_action': {...},        # Policy output
    'applied_action': {...},    # Applied action (use this!)
    'max_steps': 336,           # Episode length (from curriculum)
    ...
}
```

## Assertions

The environment includes assertions to catch schema mismatches:

```python
assert obs.shape == env.observation_space.shape
assert len(env.OBS_KEYS) == obs.shape[0]
```

If these fail, there is a schema drift bug that must be fixed.

## Curriculum Learning

When curriculum learning is enabled, `env.reset(options={'curriculum': {...}})` accepts:

- `episode_length_days`: Sets `max_steps = days * 24`
- `env.noise_temp`, `env.noise_moisture`, etc.: Noise parameters
- `actuator_limits`: Limits on actuator ranges

The environment applies these settings and logs `info['max_steps']` for diagnostics.

