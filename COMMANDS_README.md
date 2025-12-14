# Hydrosprout Digital Twin - Complete Command Reference

This document provides comprehensive documentation for all commands and capabilities of the Hydrosprout Digital Twin RL environment.

## 📋 Table of Contents

- [Quick Reference](#quick-reference)
- [Main Orchestrator Commands](#main-orchestrator-commands)
- [RL Training Commands](#rl-training-commands)
- [Evaluation Commands](#evaluation-commands)
- [Visualization & Analysis](#visualization--analysis)
- [Configuration Files](#configuration-files)
- [Schema Documentation](#schema-documentation)
- [Common Workflows](#common-workflows)
- [Troubleshooting](#troubleshooting)

---

## Quick Reference

### Most Common Commands

```bash
# Quick demo training (2k steps)
python rl/train_ppo.py --demo

# Full training (300k steps with curriculum)
python rl/train_ppo.py --timesteps 300000 --use_curriculum

# Run 14-day simulation
python main.py sim_run --steps 336

# Evaluate trained policy
python rl/evaluate_policy.py --model ppo_best/best_model.zip --n_episodes 20

# Analyze simulation logs
python main.py analyze_sim_logs --log_file logs/sim_run_*.log
```

---

## Main Orchestrator Commands

The `main.py` script provides a unified interface for all major operations.

### 1. Generate Synthetic Images

**Command:** `python main.py gen_synth`

Generate synthetic plant images for training computer vision models.

**Options:**
- `--out DIR`: Output directory (default: `synth_images`)
- `--n N`: Number of images to generate (default: 200)

**Examples:**
```bash
# Generate 300 synthetic images
python main.py gen_synth --out synth_images --n 300

# Generate 50 images for testing
python main.py gen_synth --out test_images --n 50
```

**Output:**
- Directory with numbered PNG images (`synth_0000.png`, `synth_0001.png`, ...)

---

### 2. Demo PPO Training

**Command:** `python main.py demo_ppo`

Run a quick PPO training demo (2000 steps) to verify the environment works.

**No options required.**

**Example:**
```bash
python main.py demo_ppo
```

**Output:**
- Trained model saved as `ppo_demo.zip`
- Training logs in console

---

### 3. Train Classifier

**Command:** `python main.py train_classifier`

Train a plant stage classifier on synthetic or real images.

**Options:**
- `--data DIR`: Data directory with ImageFolder layout (default: `synth_images`)
- `--epochs N`: Number of training epochs (default: 5)

**Example:**
```bash
# Train on synthetic images
python main.py train_classifier --data synth_images --epochs 10

# Train on custom dataset
python main.py train_classifier --data my_dataset --epochs 20
```

**Output:**
- Trained classifier model (typically `classifier.pth`)

---

### 4. Run Simulation

**Command:** `python main.py sim_run`

Run a simulation using either a trained PPO model or a naive/heuristic policy.

**Options:**
- `--steps N`: Number of timesteps (hours) to simulate (default: 48)
- `--species NAME`: Plant species preset (kale, broccoli, radish, etc.)
- `--model PATH`: Path to trained PPO model (default: auto-detect best model)
- `--use_extended_obs`: Use extended observations (must match training config)
- `--dose_nutrients`: Apply initial nutrient dose (N=0.1, P=0.05, K=0.1)
- `--dose_N FLOAT`: Custom nitrogen dose (0.0 to 1.0)
- `--dose_P FLOAT`: Custom phosphorus dose (0.0 to 1.0)
- `--dose_K FLOAT`: Custom potassium dose (0.0 to 1.0)
- `--no_model`: Force use of naive policy (ignore trained models)

**Examples:**
```bash
# Run 14-day simulation (336 hours) with auto-detected best model
python main.py sim_run --steps 336

# Run with specific species
python main.py sim_run --steps 336 --species kale

# Run with custom nutrient dosing
python main.py sim_run --steps 336 --dose_N 0.15 --dose_P 0.08 --dose_K 0.12

# Run with naive policy (no AI)
python main.py sim_run --steps 336 --no_model

# Run with specific model
python main.py sim_run --steps 336 --model ppo_checkpoints/ppo_model_100000_steps.zip

# Run with extended observations (if model was trained with --extended)
python main.py sim_run --steps 336 --use_extended_obs
```

**Output:**
- Log file: `logs/sim_run_YYYYMMDD_HHMMSS.log`
- Console output with step-by-step status

**Log File Contents:**
- Initial state (plant, environment, hardware)
- Per-step observations (using OBS_KEYS schema)
- Actions taken (applied_action, not raw policy output)
- Rewards and cumulative metrics
- Termination reasons if episode ends early

---

### 5. Analyze Simulation Logs

**Command:** `python main.py analyze_sim_logs`

Parse simulation log files and generate visualizations, CSV data, and interactive dashboards.

**Options:**
- `--log_file PATH`: Path to log file (default: auto-detect newest in `logs/`)
- `--out_dir DIR`: Output directory (default: `viz_output`)
- `--plot_dir DIR`: Plot directory (default: `viz_output/plots`)

**Examples:**
```bash
# Analyze most recent log file
python main.py analyze_sim_logs

# Analyze specific log file
python main.py analyze_sim_logs --log_file logs/sim_run_20251213_170448.log

# Custom output directory
python main.py analyze_sim_logs --out_dir my_analysis --plot_dir my_analysis/plots
```

**Output:**
- `parsed_sim.csv`: Time-indexed numeric time series
- `plots/*.png`: Time-series plots and diagnostics
- `summary.txt`: Numeric summary statistics
- `dashboard.html`: Interactive Plotly dashboard

---

### 6. Sample Visual

**Command:** `python main.py sample_visual`

Generate a single synthetic image sample, optionally composited with a real image.

**Options:**
- `--out PATH`: Output file path (default: `sample_vis.png`)
- `--real PATH`: Path to real image for compositing (optional)

**Examples:**
```bash
# Generate simple synthetic sample
python main.py sample_visual --out sample.png

# Composite with real image
python main.py sample_visual --out composite.png --real my_photo.jpg
```

**Output:**
- Single PNG image file

---

## RL Training Commands

### Train PPO Agent

**Command:** `python rl/train_ppo.py`

Train a Proximal Policy Optimization (PPO) agent to control the digital twin environment.

**Options:**
- `--timesteps N`: Total training timesteps (default: 300000)
- `--demo`: Quick demo training (2000 steps, smaller batch size)
- `--extended`: Use extended observations (LAI, biomass, transpiration, RH)
- `--include_soil_obs`: Include soil metrics (pH, EC, N, P, K, Fe) in observations
- `--include_nutrient_actions`: Enable nutrient dosing actions (N, P, K, pH)
- `--use_curriculum`: Enable curriculum learning (progressive difficulty)
- `--no_wrappers`: Disable observation/action normalization wrappers (not recommended)
- `--config PATH`: Path to custom config YAML file
- `--log_dir DIR`: Directory for training logs and diagnostics (default: `./training_logs`)

**Examples:**
```bash
# Quick demo (2k steps)
python rl/train_ppo.py --demo

# Basic training (200k steps)
python rl/train_ppo.py --timesteps 200000

# Full training with curriculum learning
python rl/train_ppo.py --timesteps 300000 --use_curriculum

# Training with extended observations
python rl/train_ppo.py --timesteps 200000 --extended

# Training with soil metrics
python rl/train_ppo.py --timesteps 200000 --include_soil_obs

# Training with nutrient control
python rl/train_ppo.py --timesteps 200000 --include_nutrient_actions

# Combined: extended obs + soil + nutrients + curriculum
python rl/train_ppo.py --timesteps 300000 --extended --include_soil_obs --include_nutrient_actions --use_curriculum

# Custom config file
python rl/train_ppo.py --timesteps 200000 --config configs/custom_config.yaml

# Custom log directory
python rl/train_ppo.py --timesteps 200000 --log_dir ./my_training_logs
```

**Output:**
- **Best model**: `ppo_best/best_model.zip` (saved during evaluation)
- **Final model**: `ppo_full.zip` (saved at end of training)
- **Checkpoints**: `ppo_checkpoints/ppo_model_*_steps.zip` (periodic saves)
- **TensorBoard logs**: `ppo_tensorboard/PPO_*/` (view with `tensorboard --logdir ppo_tensorboard`)
- **Evaluation logs**: `ppo_logs/evaluations.npz`
- **Training diagnostics**: `training_logs/training_progress.png` and `training_logs/summary.json`

**Training Diagnostics:**
The training script automatically generates comprehensive plots every 100 episodes:
- Episode rewards and lengths
- Plant growth and biomass
- Stress levels (water, temperature, nutrient)
- Environment conditions (temperature, humidity, CO2)
- Nutrient levels (EC, pH, N concentration)
- Hardware controls (shield, fan, Peltier modules)
- Action usage patterns
- Death rates and termination reasons
- Water use efficiency

**Monitor Training:**
```bash
# Start TensorBoard
tensorboard --logdir ppo_tensorboard

# Open browser to http://localhost:6006
```

---

## Evaluation Commands

### Evaluate Trained Policy

**Command:** `python rl/evaluate_policy.py`

Run deterministic evaluation of a trained policy with fixed seeds for reproducibility.

**Options:**
- `--model PATH`: Path to trained PPO model (required)
- `--n_episodes N`: Number of episodes to run (default: 20)
- `--seed N`: Random seed for reproducibility (default: 42)
- `--deterministic`: Use deterministic policy (default: True)
- `--stochastic`: Use stochastic policy (overrides --deterministic)

**Examples:**
```bash
# Evaluate best model (20 episodes, deterministic)
python rl/evaluate_policy.py --model ppo_best/best_model.zip

# Evaluate with more episodes
python rl/evaluate_policy.py --model ppo_best/best_model.zip --n_episodes 50

# Evaluate with different seed
python rl/evaluate_policy.py --model ppo_best/best_model.zip --seed 123

# Evaluate with stochastic policy
python rl/evaluate_policy.py --model ppo_best/best_model.zip --stochastic

# Evaluate checkpoint
python rl/evaluate_policy.py --model ppo_checkpoints/ppo_model_200000_steps.zip
```

**Output:**
- Console summary with:
  - Mean reward ± std
  - Mean episode length ± std
  - Mean biomass gain ± std
  - Mean stress levels (water, temp, nutrient)
  - % of episodes reaching max_steps
  - Termination reason distribution
  - Sample trajectory (first 5 steps) with true_state, observed_state, and applied_action

**Example Output:**
```
================================================================================
EVALUATION SUMMARY
================================================================================
Episodes: 20
Mean Reward: 1250.34 ± 45.67
Mean Episode Length: 336.0 ± 0.0 steps
Mean Biomass Gain: 12.45 ± 2.34 g
Mean Stresses: Water=0.923, Temp=0.987, Nutrient=0.945
% Reaching Max Steps: 100.0%

Termination Reasons:
  completed: 20 (100.0%)
================================================================================
```

---

## Visualization & Analysis

### Training Progress Plots

Training automatically generates plots in `training_logs/training_progress.png`:
- Episode rewards (smoothed and raw)
- Episode lengths (survival time)
- Cumulative biomass gain
- Plant state features (biomass, moisture, LAI, height, NSC)
- Plant stress levels (water, temperature, nutrient)
- Environment conditions (temperature zones, humidity zones, CO2)
- Nutrient levels (EC, pH, N concentration)
- Hardware controls (shield, fan, Peltier modules)
- Water and coverage efficiency
- Action usage over time
- Death rates
- Water use efficiency

### Simulation Log Analysis

The `analyze_sim_logs` command generates:
- **Time-series plots**: All state variables over time
- **CSV data**: Machine-readable time series
- **Interactive dashboard**: Plotly HTML dashboard with zoom/pan
- **Summary statistics**: Mean, std, min, max for all variables

### TensorBoard

View real-time training metrics:
```bash
tensorboard --logdir ppo_tensorboard
```

Available metrics:
- Policy loss
- Value function loss
- Entropy
- Explained variance
- Episode reward
- Episode length
- Learning rate

---

## Configuration Files

### Default Configuration

Location: `config/defaults.yaml`

Contains default parameters for:
- PPO hyperparameters (learning rate, batch size, etc.)
- Reward function weights
- Environment parameters
- Plant model parameters

### Species Configuration

Location: `config/species_config.yaml`

Plant species presets with parameters for:
- Kale
- Broccoli
- Radish
- (Add more as needed)

**Usage:**
```bash
# Use species preset in simulation
python main.py sim_run --steps 336 --species kale
```

### Custom Configuration

Create a custom config file and reference it:
```bash
python rl/train_ppo.py --timesteps 200000 --config configs/my_config.yaml
```

---

## Schema Documentation

### Observation Schema

The observation vector follows a strict schema defined by `OBS_KEYS`. **Never hardcode indices!**

**Full Documentation:** See `OBSERVATION_ACTION_SCHEMA.md`

**Quick Reference:**
- 11 plant features (biomass, moisture, nutrient, LAI, stresses, height, NSC, N_content, total_biomass)
- 8 environment features (3 temp zones, temp_stress, 3 RH zones, CO2)
- 5 nutrient features (EC, pH, N_ppm, EC_stress, pH_stress)
- 5 + n_peltiers hardware features (shield, fan, moisture_std, coverage_efficiency, water_efficiency, Peltier states)
- 2 time features (hour sin, hour cos)

**Total:** 31 + n_peltiers dimensions (typically 35)

**Decoding Observations:**
```python
# Correct way (use schema)
obs_keys = env.OBS_KEYS
obs_dict = {key: obs[i] for i, key in enumerate(obs_keys)}
temp = obs_dict['env_T_middle'] * 10.0 + 25.0  # Denormalize

# Wrong way (hardcoded indices)
temp = obs[13] * 10.0 + 25.0  # DON'T DO THIS
```

### Action Schema

Actions are provided as a Dict space with scaling factors:

- `water_total`: [0, 1] → 0.05 L per unit
- `fan`: [0, 1] → binary (on/off)
- `shield_delta`: [-1, 1] → 0.2 per unit
- `heater`: [0, 1] → 200 W per unit
- `peltier_controls`: [-1, 1] array → direct power control
- `dose_N`: [0, 1] → 0.5 g per unit
- `dose_P`: [0, 1] → 0.1 g per unit
- `dose_K`: [0, 1] → 0.3 g per unit
- `pH_adjust`: [-1, 1] → pH adjustment
- `nozzle_mask`: [0, 1] array → binary nozzle activation
- `co2_inject`: [0, 1] → 10.0 L/hour per unit

**Action Logging:**
The environment logs both `raw_action` (policy output) and `applied_action` (after scaling/clipping). **Always use `applied_action` for diagnostics!**

---

## Common Workflows

### Workflow 1: Quick Test

```bash
# 1. Quick demo training
python rl/train_ppo.py --demo

# 2. Run short simulation
python main.py sim_run --steps 48

# 3. Analyze results
python main.py analyze_sim_logs
```

### Workflow 2: Full Training and Evaluation

```bash
# 1. Train with curriculum learning
python rl/train_ppo.py --timesteps 300000 --use_curriculum

# 2. Evaluate best model
python rl/evaluate_policy.py --model ppo_best/best_model.zip --n_episodes 20

# 3. Run 14-day simulation
python main.py sim_run --steps 336

# 4. Analyze simulation
python main.py analyze_sim_logs
```

### Workflow 3: Species-Specific Training

```bash
# 1. Train for specific species
python rl/train_ppo.py --timesteps 200000 --config configs/kale_config.yaml

# 2. Run simulation with species preset
python main.py sim_run --steps 336 --species kale

# 3. Compare with baseline
python main.py sim_run --steps 336 --species kale --no_model
```

### Workflow 4: Advanced Training with All Features

```bash
# 1. Train with extended observations, soil metrics, nutrient control, and curriculum
python rl/train_ppo.py --timesteps 300000 \
    --extended \
    --include_soil_obs \
    --include_nutrient_actions \
    --use_curriculum

# 2. Evaluate
python rl/evaluate_policy.py --model ppo_best/best_model.zip --n_episodes 50

# 3. Run simulation with nutrient dosing
python main.py sim_run --steps 336 --dose_N 0.15 --dose_P 0.08 --dose_K 0.12
```

### Workflow 5: Debugging and Analysis

```bash
# 1. Run simulation with detailed logging
python main.py sim_run --steps 336 --no_model > debug.log 2>&1

# 2. Analyze logs
python main.py analyze_sim_logs --log_file logs/sim_run_*.log

# 3. Check training diagnostics
# View training_logs/training_progress.png
# View training_logs/summary.json
```

---

## Troubleshooting

### Common Issues

#### 1. "Observation shape mismatch" error

**Cause:** Schema drift between training and evaluation.

**Solution:**
- Ensure `--use_extended_obs` flag matches training configuration
- Check that observation space dimensions match (use `env.observation_space.shape`)
- Verify `OBS_KEYS` length matches observation vector length

#### 2. "Model not found" error

**Cause:** No trained model available or wrong path.

**Solution:**
```bash
# Check available models
ls ppo_best/
ls ppo_checkpoints/

# Use explicit path
python main.py sim_run --steps 336 --model ppo_best/best_model.zip

# Or use --no_model for naive policy
python main.py sim_run --steps 336 --no_model
```

#### 3. Training crashes or NaN values

**Cause:** Numerical instability or extreme observations.

**Solution:**
- Check `training_logs/training_progress.png` for anomalies
- Reduce learning rate in config
- Enable observation normalization (don't use `--no_wrappers`)
- Check for thermal blow-ups (should be fixed with recent updates)

#### 4. Low survival rate

**Cause:** Policy not learning or environment too difficult.

**Solution:**
- Use curriculum learning: `--use_curriculum`
- Increase training timesteps
- Check reward function in config
- Verify termination conditions are reasonable

#### 5. Coverage efficiency always 0.0

**Cause:** Coordinate system mismatch between plants and nozzles.

**Solution:**
- This should be auto-fixed in recent updates
- Check plant and nozzle coordinates are in meters (not normalized)
- Verify nozzle radius is reasonable (typically 0.1m)

#### 6. Action values seem wrong

**Cause:** Using raw_action instead of applied_action.

**Solution:**
- Always use `info['applied_action']` for diagnostics
- Check `ACTION_KEYS` for scaling factors
- Verify action space matches model training

### Getting Help

1. **Check logs**: Review `logs/sim_run_*.log` for detailed error messages
2. **Check diagnostics**: View `training_logs/training_progress.png` for training issues
3. **Verify schema**: Use `info['obs_keys']` and `info['action_keys']` to verify schemas
4. **Run evaluation**: Use `evaluate_policy.py` to test model in isolation
5. **Check assertions**: Environment includes assertions to catch schema drift

### Debug Mode

Enable verbose logging:
```bash
# Python logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or check info dict
info = env.step(action)
print("OBS_KEYS:", info['obs_keys'])
print("True state:", info['true_state'])
print("Observed state:", info['observed_state'])
print("Applied action:", info['applied_action'])
```

---

## Additional Resources

- **Schema Documentation**: `OBSERVATION_ACTION_SCHEMA.md` - Complete observation/action schema reference
- **Training Diagnostics**: `training_logs/training_progress.png` - Comprehensive training plots
- **TensorBoard**: `ppo_tensorboard/` - Real-time training metrics
- **Simulation Logs**: `logs/sim_run_*.log` - Detailed step-by-step simulation logs
- **Analysis Output**: `viz_output/` - Parsed data, plots, and dashboards

---

## Command Summary Table

| Command | Purpose | Key Options |
|---------|---------|-------------|
| `python main.py gen_synth` | Generate synthetic images | `--out`, `--n` |
| `python main.py demo_ppo` | Quick PPO demo | None |
| `python main.py train_classifier` | Train classifier | `--data`, `--epochs` |
| `python main.py sim_run` | Run simulation | `--steps`, `--model`, `--species`, `--dose_*` |
| `python main.py analyze_sim_logs` | Analyze logs | `--log_file`, `--out_dir` |
| `python main.py sample_visual` | Generate sample image | `--out`, `--real` |
| `python rl/train_ppo.py` | Train PPO agent | `--timesteps`, `--use_curriculum`, `--extended`, etc. |
| `python rl/evaluate_policy.py` | Evaluate policy | `--model`, `--n_episodes`, `--seed` |

---

**Last Updated:** 2024-12-13  
**Version:** 2.0 (with comprehensive fixes and schema documentation)

