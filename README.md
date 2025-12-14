# Hydrosprout Digital Twin

A reinforcement learning-based digital twin system for automated microgreens cultivation. This project uses Proximal Policy Optimization (PPO) to train an AI agent that controls environmental conditions (temperature, moisture, light, nutrients) to optimize plant growth while preventing plant death, mold, and resource waste.

## 🌱 Features

- **Plant Growth Simulation**: Physics-based plant growth model with structural-functional plant modeling (SFPM)
- **Reinforcement Learning**: PPO agent trained to optimize growing conditions
- **Multi-Model Integration**: Combines plant biology, soil chemistry, environmental physics, and hardware control
- **Curriculum Learning**: Progressive difficulty training for robust policy learning
- **Comprehensive Monitoring**: Real-time tracking of canopy growth, biomass, moisture, temperature, nutrients, and mold risk
- **Death Prevention**: Strong penalties and termination conditions to prevent plant death from:
  - Extreme temperatures (cooked >40°C or frozen <5°C)
  - Overwatering (waterlogging >95% moisture)
  - Underwatering (dried out <10% moisture)
  - Biomass collapse (below minimum viable threshold)
- **Reward Shaping**: Balanced reward function with:
  - Canopy growth rewards
  - Moisture/temperature range optimization
  - Mold prevention penalties
  - Energy efficiency considerations
  - Plant death penalties (-1000.0)

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training PPO Agent](#training-ppo-agent)
- [Running Simulations](#running-simulations)
- [Complete Command Reference](#complete-command-reference)
- [Visualization & Analysis](#visualization--analysis)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## 📚 Documentation

- **[COMMANDS_README.md](COMMANDS_README.md)** - Complete command reference with all options and examples
- **[OBSERVATION_ACTION_SCHEMA.md](OBSERVATION_ACTION_SCHEMA.md)** - Detailed observation and action schema documentation
- [Complete Command Reference](#complete-command-reference)

## 📚 Documentation

- **[COMMANDS_README.md](COMMANDS_README.md)** - Complete command reference with all options and examples
- **[OBSERVATION_ACTION_SCHEMA.md](OBSERVATION_ACTION_SCHEMA.md)** - Detailed observation and action schema documentation

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip
- (Optional) CUDA-capable GPU for faster training

### Setup

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd digitaltwin
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD)**:
     ```cmd
     venv\Scripts\activate.bat
     ```
   - **Linux/Mac**:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r Requirements.txt
   ```

   Or install with extra dependencies for stable-baselines3:
   ```bash
   pip install stable-baselines3[extra] gymnasium torch
   ```

## 📁 Project Structure

```
digitaltwin/
├── config/                 # Configuration files
│   ├── defaults.yaml      # Default PPO and reward parameters
│   └── species_config.yaml # Plant species presets (kale, broccoli, radish)
├── rl/                    # Reinforcement learning components
│   ├── gym_env.py         # Gymnasium environment wrapper
│   ├── train_ppo.py       # PPO training script
│   ├── wrappers.py        # Observation/action normalization wrappers
│   └── curriculum.py      # Curriculum learning scheduler
├── sim/                   # Simulation models
│   ├── plant.py           # Simple plant model
│   ├── plant_SFPM.py      # Structural-functional plant model
│   ├── plant_adapter.py   # Adapter for PlantStructural
│   ├── env_model.py       # Environmental dynamics (temp, light, RH)
│   ├── hardware.py        # Hardware control (pump, fan, heater, shield)
│   ├── sensors.py         # Sensor noise and drift models
│   └── soil_model.py      # Soil chemistry (pH, EC, macro/micronutrients)
├── visionmodel/           # Computer vision components
│   ├── synth_generator.py # Synthetic image generation
│   └── train_classifier.py # Plant stage classifier training
├── viz/                   # Visualization tools
│   ├── analyze_sim_logs.py # Parse and analyze simulation logs
│   ├── animate.py         # Create animations from logs
│   └── plot_utils.py      # Plotting utilities
├── tests/                 # Unit tests
├── notebooks/            # Jupyter notebooks
├── main.py               # Main orchestrator script
├── config.py             # Config loading utilities
└── Requirements.txt      # Python dependencies
```

## 🎯 Quick Start

### 1. Run a Quick Demo

Train a PPO agent for 2000 steps (quick test):
```bash
python rl/train_ppo.py --demo
```

### 2. Run a Simulation

Run a simulation with a trained model (or naive policy if no model):
```bash
python main.py sim_run --steps 336  # 14 days (336 hours)
```

### 3. Analyze Results

Analyze simulation logs and generate visualizations:
```bash
python main.py analyze_sim_logs --log_file logs/sim_run_*.log
```

## 🎓 Training PPO Agent

### Basic Training

Train a PPO agent with default settings:
```bash
python rl/train_ppo.py --timesteps 200000
```

### Training Options

```bash
python rl/train_ppo.py [OPTIONS]
```

**Options:**
- `--timesteps N`: Total training timesteps (default: 200000)
- `--demo`: Quick demo training (2000 steps)
- `--extended`: Use extended observations (LAI, biomass, transpiration, RH)
- `--include_soil_obs`: Include soil metrics (pH, EC, N, P, K, Fe) in observations
- `--include_nutrient_actions`: Add nutrient dosing actions (N, P, K)
- `--use_curriculum`: Enable curriculum learning (progressive difficulty)
- `--no_wrappers`: Disable observation/action normalization wrappers
- `--config PATH`: Path to custom config YAML file

### Examples

**Full training with curriculum learning:**
```bash
python rl/train_ppo.py --timesteps 300000 --use_curriculum
```

**Training with extended observations and soil metrics:**
```bash
python rl/train_ppo.py --timesteps 200000 --extended --include_soil_obs
```

**Training with nutrient control:**
```bash
python rl/train_ppo.py --timesteps 200000 --include_nutrient_actions
```

### Training Outputs

- **Best model**: `ppo_best/best_model/` (saved during evaluation)
- **Checkpoints**: `ppo_checkpoints/ppo_model_*_steps.zip` (periodic saves)
- **TensorBoard logs**: `ppo_tensorboard/PPO_*/` (view with `tensorboard --logdir ppo_tensorboard`)
- **Evaluation logs**: `ppo_logs/evaluations.npz`

### Monitor Training

View training progress with TensorBoard:
```bash
tensorboard --logdir ppo_tensorboard
```

Then open `http://localhost:6006` in your browser.

## 📚 Complete Command Reference

For comprehensive documentation of all commands, options, and examples, see:

**[COMMANDS_README.md](COMMANDS_README.md)** - Complete command reference with:
- All main.py commands (gen_synth, demo_ppo, train_classifier, sim_run, analyze_sim_logs, sample_visual)
- All rl/train_ppo.py training options
- Evaluation commands (evaluate_policy.py)
- Common workflows and troubleshooting
- Schema documentation reference

## 🧪 Running Simulations

### Using Main Script

```bash
python main.py sim_run [OPTIONS]
```

**Options:**
- `--steps N`: Number of timesteps to simulate (default: 48)
- `--model PATH`: Path to trained PPO model (default: auto-detect best model)
- `--use_extended_obs`: Use extended observations (must match training)
- `--dose_nutrients`: Apply initial nutrient dose (N=0.1, P=0.05, K=0.1)
- `--dose_N FLOAT`: Custom nitrogen dose (0..1)
- `--dose_P FLOAT`: Custom phosphorus dose (0..1)
- `--dose_K FLOAT`: Custom potassium dose (0..1)
- `--no_model`: Use naive policy instead of trained model

### Examples

**Run 14-day simulation with trained model:**
```bash
python main.py sim_run --steps 336
```

**Run with custom nutrient dosing:**
```bash
python main.py sim_run --steps 336 --dose_N 0.15 --dose_P 0.08 --dose_K 0.12
```

**Run with naive policy (no AI):**
```bash
python main.py sim_run --steps 336 --no_model
```

### Simulation Outputs

- **Log files**: `logs/sim_run_YYYYMMDD_HHMMSS.log` (detailed step-by-step logs)
- **Console output**: Real-time status updates

## 📊 Visualization & Analysis

### Analyze Simulation Logs

Parse and visualize simulation results:
```bash
python main.py analyze_sim_logs --log_file logs/sim_run_*.log
```

Or use the analyzer directly:
```bash
python viz/analyze_sim_logs.py logs/sim_run_*.log
```

**Outputs:**
- `viz_output/parsed_sim.csv`: Parsed data in CSV format
- `viz_output/plots/`: Generated plots:
  - `biomass_stacked.png`: Leaf, stem, root biomass over time
  - `canopy_moist_nutrient.png`: Canopy, moisture, and nutrient trends
  - `temp_lux.png`: Temperature and light conditions
  - `energy_water.png`: Energy consumption and water usage
  - `mold_prob.png`: Mold probability over time
  - `reward_curve.png`: Reward signal over time
  - `correlation.png`: Correlation matrix of variables
- `viz_output/dashboard.html`: Interactive dashboard
- `viz_output/summary.txt`: Text summary of simulation

### Generate Synthetic Images

Create synthetic plant images for computer vision tasks:
```bash
python main.py gen_synth --out synth_images --n 300
```

### Sample Visualization

Generate a sample synthetic image:
```bash
python main.py sample_visual --out sample.png
```

## ⚙️ Configuration

### Config Files

- **`config/defaults.yaml`**: Default PPO hyperparameters and reward weights
- **`config/species_config.yaml`**: Plant species-specific parameters

### Reward Configuration

Edit `config/defaults.yaml` to adjust reward weights:

```yaml
reward:
  w_canopy: 10.0      # Canopy growth reward
  w_mold: 5.0         # Mold penalty
  w_energy: 0.001     # Energy cost penalty
  w_moisture: 3.0     # Moisture range reward
  w_temp: 2.0         # Temperature range reward
```

### PPO Hyperparameters

```yaml
ppo:
  learning_rate: 1e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.995
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.02
  vf_coef: 0.5
  max_grad_norm: 0.5
```

### Environment Configuration

Key parameters in `gym_env.py`:
- `DEFAULT_MAX_WATER_L = 0.05`: Maximum liters per watering action
- `MAX_TEMP_TERMINATE = 38.0`: Temperature threshold for termination
- `OVERWATER_TERMINATE_THRESH = 0.95`: Moisture threshold for termination
- `MOISTURE_OPT = 0.50`: Optimal moisture level
- `MOISTURE_BAND = 0.15`: Acceptable moisture range
- `DEATH_PENALTY = -1000.0`: Penalty for plant death

## 🔬 Advanced Usage

### Curriculum Learning

Curriculum learning progressively increases difficulty:
- **Stage 1 (Easy)**: Optimal conditions, minimal noise
- **Stage 2 (Medium)**: Variable conditions, moderate noise
- **Stage 3 (Hard)**: Challenging conditions, high noise
- **Stage 4 (Expert)**: Realistic conditions, disturbances

Enable with `--use_curriculum` flag.

### Custom Observation Spaces

**Base observations (7 dims):**
- Canopy cover
- Moisture
- Nutrient availability
- Mold probability
- Temperature (normalized)
- Light intensity
- Shield position

**Extended observations (+5 dims):**
- LAI (Leaf Area Index)
- Leaf biomass
- Root biomass
- Transpiration rate
- Relative humidity

**Soil observations (+6 dims):**
- pH
- EC (Electrical Conductivity)
- Soil N, P, K
- Soil Fe (iron)

### Action Space

**Standard actions (4 dims):**
- Water (0..1) → scaled to liters
- Fan (0/1) → on/off
- Shield delta (-1..1) → position change
- Heater (0..1) → power level

**With nutrients (7 dims):**
- Above + N, P, K dosing (0..1 each)

### Training Tips

1. **Start with demo**: Use `--demo` to verify setup
2. **Use curriculum**: Enable `--use_curriculum` for better convergence
3. **Monitor TensorBoard**: Watch for reward collapse or policy degradation
4. **Adjust reward weights**: If agent ignores important signals, increase weights
5. **Check plant death rate**: High death rate may indicate reward imbalance

## 🐛 Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'stable_baselines3'**
```bash
pip install stable-baselines3[extra] gymnasium
```

**2. Learning rate type error**
- The code automatically handles this, but if you see errors, ensure config values are floats, not strings

**3. Environment reset errors**
- Ensure `self.cfg` is initialized as a dict, not None
- Check that curriculum configs are properly formatted

**4. Plant death too frequent**
- Increase `w_moisture` and `w_temp` in reward config
- Decrease `MAX_TEMP_TERMINATE` threshold
- Check initial conditions in config

**5. Training collapse (rewards go negative)**
- Reduce learning rate
- Increase entropy coefficient (`ent_coef`)
- Check reward weights balance
- Enable curriculum learning

### Debug Mode

Run with verbose output:
```bash
python rl/train_ppo.py --timesteps 20000 --verbose 2
```

Check simulation logs:
```bash
# View latest log
cat logs/sim_run_*.log | tail -100

# Search for errors
grep -i error logs/sim_run_*.log
```

## 📚 Additional Resources

### Key Files to Understand

- **`rl/gym_env.py`**: Environment implementation and reward function
- **`sim/plant_SFPM.py`**: Plant growth model
- **`sim/soil_model.py`**: Soil chemistry and nutrient dynamics
- **`rl/train_ppo.py`**: Training script and hyperparameters

### Testing

Run unit tests:
```bash
python -m pytest tests/
```

### Jupyter Notebooks

Explore the system interactively:
```bash
jupyter notebook notebooks/demo.ipynb
```

## 📝 Citation

If you use this project in your research, please cite:

```
Hydrosprout Digital Twin: Reinforcement Learning for Automated Microgreens Cultivation
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

[Specify your license here]

## 🙏 Acknowledgments

- Stable-Baselines3 for PPO implementation
- Gymnasium for RL environment interface
- Plant growth modeling based on structural-functional plant models

---

**Happy Growing! 🌱**

