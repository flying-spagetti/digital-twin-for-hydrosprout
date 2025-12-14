# rl/diagnostics.py
"""
Training diagnostics and visualization tools
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
import json


class TrainingDiagnostics:
    """Track and visualize agent learning progress"""
    
    def __init__(self, log_dir: str = "./training_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_growth = []
        self.episode_deaths = []
        self.episode_water_use = []
        
        # Action statistics
        self.action_history = {
            'water': [],
            'fan': [],
            'shield': [],
            'heater': []
        }
        
        # Plant state history (all observation features)
        self.plant_states = {
            'biomass': [],
            'moisture': [],
            'LAI': [],
            'temp': [],
        }
        
        # Track all observation features for comprehensive plotting
        self.observation_features = {
            # Plant features (11)
            'plant_biomass': [],
            'plant_moisture': [],
            'plant_nutrient': [],
            'plant_LAI': [],
            'plant_stress_water': [],
            'plant_stress_temp': [],
            'plant_stress_nutrient': [],
            'plant_height': [],
            'plant_NSC': [],
            'plant_N_content': [],
            'plant_total_biomass': [],
            # Environment features (8)
            'env_T_top': [],
            'env_T_middle': [],
            'env_T_bottom': [],
            'env_temp_stress': [],
            'env_RH_top': [],
            'env_RH_middle': [],
            'env_RH_bottom': [],
            'env_CO2': [],
            # Nutrient features (5)
            'nutrient_EC': [],
            'nutrient_pH': [],
            'nutrient_N_ppm': [],
            'nutrient_EC_stress': [],
            'nutrient_pH_stress': [],
            # Hardware features (5)
            'hw_shield_pos': [],
            'hw_fan_on': [],
            'hw_moisture_std': [],
            'hw_coverage_efficiency': [],
            'hw_water_efficiency': [],
            # Peltier modules (dynamic, typically 4)
            'peltier_0': [],
            'peltier_1': [],
            'peltier_2': [],
            'peltier_3': [],
        }
    
    def log_episode(self, info: Dict[str, Any], actions: List[np.ndarray], obs_features: Dict[str, float] = None):
        """Log episode statistics"""
        self.episode_rewards.append(info.get('episode_reward', 0))
        self.episode_lengths.append(info.get('episode_length', 0))
        self.episode_growth.append(info.get('cumulative_growth', 0))
        # FIXED: Count true terminations (not just death_reason presence)
        # Check both terminated and truncated flags if available
        death_reason = info.get('death_reason', None)
        terminated = info.get('terminated', False) if 'terminated' in info else (death_reason is not None)
        self.episode_deaths.append(1 if terminated else 0)
        self.episode_water_use.append(info.get('cumulative_water_used', 0))
        
        # Track termination reasons for distribution analysis
        if not hasattr(self, 'termination_reasons'):
            self.termination_reasons = []
        if death_reason:
            self.termination_reasons.append(death_reason)
        elif terminated:
            self.termination_reasons.append('truncated')
        else:
            self.termination_reasons.append('completed')
        
        # Aggregate action statistics
        # FIXED: Use applied_action from info if available (not raw policy output)
        if actions:
            # Try to get applied actions from info if available
            applied_actions = None
            if isinstance(info, dict) and 'applied_action' in info:
                applied_actions = info['applied_action']
            
            if applied_actions:
                # Use applied actions (after scaling/clipping)
                self.action_history['water'].append(applied_actions.get('water_total', 0.0))
                self.action_history['fan'].append(1.0 if applied_actions.get('fan', False) else 0.0)
                self.action_history['shield'].append(applied_actions.get('shield_delta', 0.0))
                self.action_history['heater'].append(applied_actions.get('heater', 0.0))
            else:
                # Fallback: use raw actions (may be from old code)
                actions_array = np.array(actions)
                if actions_array.shape[1] >= 4:
                    self.action_history['water'].append(actions_array[:, 0].mean())
                    self.action_history['fan'].append(actions_array[:, 1].mean())
                    self.action_history['shield'].append(actions_array[:, 2].mean())
                    self.action_history['heater'].append(actions_array[:, 3].mean())
        
        # Log observation features (mean over episode)
        if obs_features:
            for key, value in obs_features.items():
                if key in self.observation_features:
                    self.observation_features[key].append(value)
                else:
                    # Initialize if not present
                    self.observation_features[key] = [value]
    
    def log_step(self, plant_state: Dict[str, Any], env_state: Dict[str, Any]):
        """Log individual step for detailed analysis"""
        self.plant_states['biomass'].append(plant_state.get('biomass_total', 0))
        self.plant_states['moisture'].append(plant_state.get('soil_moisture', 0))
        self.plant_states['LAI'].append(plant_state.get('LAI', 0))
        self.plant_states['temp'].append(env_state.get('T', 20))
    
    def plot_training_progress(self, window: int = 100):
        """Plot comprehensive training metrics including all observation features"""
        # Create a larger figure with multiple subplots
        # 7 rows x 3 columns = 21 plots (we'll use 18-19)
        fig = plt.figure(figsize=(20, 28))
        gs = fig.add_gridspec(7, 3, hspace=0.3, wspace=0.3)
        fig.suptitle('Training Progress - All Features', fontsize=18, y=0.995)
        
        # Helper function to smooth data
        def smooth_data(data, window_size):
            if len(data) > window_size:
                return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
            return data
        
        # Row 0: Core Metrics
        # 1. Episode Rewards
        ax = fig.add_subplot(gs[0, 0])
        if len(self.episode_rewards) > window:
            smoothed = smooth_data(self.episode_rewards, window)
            ax.plot(smoothed, label=f'Smoothed (window={window})', linewidth=2)
        ax.plot(self.episode_rewards, alpha=0.3, label='Raw', linewidth=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Episode Lengths
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(self.episode_lengths, linewidth=1)
        ax.axhline(y=336, color='r', linestyle='--', label='Max (14 days)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Length (Survival)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Cumulative Growth
        ax = fig.add_subplot(gs[0, 2])
        if len(self.episode_growth) > window:
            smoothed = smooth_data(self.episode_growth, window)
            ax.plot(smoothed, label=f'Smoothed', linewidth=2)
        ax.plot(self.episode_growth, alpha=0.3, label='Raw', linewidth=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Growth (g)')
        ax.set_title('Cumulative Biomass Gain')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Row 1: Plant State Features
        # 4. Plant Biomass & Moisture
        ax = fig.add_subplot(gs[1, 0])
        ax2_moisture = None
        if self.observation_features.get('plant_biomass'):
            data = self.observation_features['plant_biomass']
            if len(data) > window:
                smoothed = smooth_data(data, window)
                ax.plot(smoothed, label='Biomass', linewidth=2)
        if self.observation_features.get('plant_moisture'):
            data = self.observation_features['plant_moisture']
            if len(data) > window:
                smoothed = smooth_data(data, window)
                ax2_moisture = ax.twinx()
                ax2_moisture.plot(smoothed, label='Moisture', color='blue', alpha=0.7, linewidth=2)
                ax2_moisture.set_ylabel('Moisture', color='blue')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Biomass')
        ax.set_title('Plant Biomass & Moisture')
        ax.legend(loc='upper left')
        if ax2_moisture is not None:
            ax2_moisture.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 5. Plant Stresses
        ax = fig.add_subplot(gs[1, 1])
        for stress_type in ['plant_stress_water', 'plant_stress_temp', 'plant_stress_nutrient']:
            if self.observation_features.get(stress_type):
                data = self.observation_features[stress_type]
                if len(data) > window:
                    smoothed = smooth_data(data, window)
                    label = stress_type.replace('plant_stress_', '').title()
                    ax.plot(smoothed, label=label, linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Stress Factor')
        ax.set_title('Plant Stress Levels')
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
        
        # 6. Plant LAI & Height
        ax = fig.add_subplot(gs[1, 2])
        ax2_height = None
        if self.observation_features.get('plant_LAI'):
            data = self.observation_features['plant_LAI']
            if len(data) > window:
                smoothed = smooth_data(data, window)
                ax.plot(smoothed, label='LAI', linewidth=2)
        if self.observation_features.get('plant_height'):
            data = self.observation_features['plant_height']
            if len(data) > window:
                smoothed = smooth_data(data, window)
                ax2_height = ax.twinx()
                ax2_height.plot(smoothed, label='Height', color='green', alpha=0.7, linewidth=2)
                ax2_height.set_ylabel('Height (m)', color='green')
        ax.set_xlabel('Episode')
        ax.set_ylabel('LAI')
        ax.set_title('Plant LAI & Height')
        ax.legend(loc='upper left')
        if ax2_height is not None:
            ax2_height.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Row 2: Environment Features
        # 7. Temperature Zones
        ax = fig.add_subplot(gs[2, 0])
        for temp_zone in ['env_T_top', 'env_T_middle', 'env_T_bottom']:
            if self.observation_features.get(temp_zone):
                data = self.observation_features[temp_zone]
                if len(data) > window:
                    smoothed = smooth_data(data, window)
                    # Denormalize: (T-25)/10, so T = smoothed*10 + 25
                    temp_C = smoothed * 10 + 25
                    label = temp_zone.replace('env_T_', '').title()
                    ax.plot(temp_C, label=label, linewidth=2)
        ax.axhline(y=25, color='g', linestyle='--', alpha=0.5, label='Optimal')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature Zones')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 8. Relative Humidity
        ax = fig.add_subplot(gs[2, 1])
        for rh_zone in ['env_RH_top', 'env_RH_middle', 'env_RH_bottom']:
            if self.observation_features.get(rh_zone):
                data = self.observation_features[rh_zone]
                if len(data) > window:
                    smoothed = smooth_data(data, window)
                    # Denormalize: RH/100, so RH = smoothed*100
                    rh_pct = smoothed * 100
                    label = rh_zone.replace('env_RH_', '').title()
                    ax.plot(rh_pct, label=label, linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Relative Humidity (%)')
        ax.set_title('Humidity Zones')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 9. CO2 & Temp Stress
        ax = fig.add_subplot(gs[2, 2])
        ax2_co2 = None
        if self.observation_features.get('env_CO2'):
            data = self.observation_features['env_CO2']
            if len(data) > window:
                smoothed = smooth_data(data, window)
                # Denormalize: CO2/2000, so CO2 = smoothed*2000
                co2_ppm = smoothed * 2000
                ax.plot(co2_ppm, label='CO2', linewidth=2, color='purple')
        if self.observation_features.get('env_temp_stress'):
            data = self.observation_features['env_temp_stress']
            if len(data) > window:
                smoothed = smooth_data(data, window)
                ax2_co2 = ax.twinx()
                ax2_co2.plot(smoothed, label='Temp Stress', color='red', alpha=0.7, linewidth=2)
                ax2_co2.set_ylabel('Temp Stress', color='red')
        ax.set_xlabel('Episode')
        ax.set_ylabel('CO2 (ppm)')
        ax.set_title('CO2 & Temperature Stress')
        ax.legend(loc='upper left')
        if ax2_co2 is not None:
            ax2_co2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Row 3: Nutrient Features
        # 10. EC & pH
        ax = fig.add_subplot(gs[3, 0])
        ax2_ph = None
        if self.observation_features.get('nutrient_EC'):
            data = self.observation_features['nutrient_EC']
            if len(data) > window:
                smoothed = smooth_data(data, window)
                # Denormalize: EC/3, so EC = smoothed*3
                ec_value = smoothed * 3
                ax.plot(ec_value, label='EC', linewidth=2, color='blue')
        if self.observation_features.get('nutrient_pH'):
            data = self.observation_features['nutrient_pH']
            if len(data) > window:
                smoothed = smooth_data(data, window)
                # Denormalize: (pH-4)/4, so pH = smoothed*4 + 4
                ph_value = smoothed * 4 + 4
                ax2_ph = ax.twinx()
                ax2_ph.plot(ph_value, label='pH', color='orange', linewidth=2)
                ax2_ph.set_ylabel('pH', color='orange')
        ax.set_xlabel('Episode')
        ax.set_ylabel('EC (mS/cm)')
        ax.set_title('Nutrient EC & pH')
        ax.legend(loc='upper left')
        if ax2_ph is not None:
            ax2_ph.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 11. N Concentration
        ax = fig.add_subplot(gs[3, 1])
        if self.observation_features.get('nutrient_N_ppm'):
            data = self.observation_features['nutrient_N_ppm']
            if len(data) > window:
                smoothed = smooth_data(data, window)
                # Denormalize: N_ppm/100, so N_ppm = smoothed*100
                n_ppm = smoothed * 100
                ax.plot(n_ppm, label='N (ppm)', linewidth=2, color='green')
        ax.set_xlabel('Episode')
        ax.set_ylabel('N Concentration (ppm)')
        ax.set_title('Nitrogen Concentration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 12. Nutrient Stresses
        ax = fig.add_subplot(gs[3, 2])
        for stress_type in ['nutrient_EC_stress', 'nutrient_pH_stress']:
            if self.observation_features.get(stress_type):
                data = self.observation_features[stress_type]
                if len(data) > window:
                    smoothed = smooth_data(data, window)
                    label = stress_type.replace('nutrient_', '').replace('_stress', '').upper() + ' Stress'
                    ax.plot(smoothed, label=label, linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Stress Factor')
        ax.set_title('Nutrient Stress Levels')
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
        
        # Row 4: Hardware & Actions
        # 13. Hardware Controls
        ax = fig.add_subplot(gs[4, 0])
        if self.observation_features.get('hw_shield_pos'):
            data = self.observation_features['hw_shield_pos']
            if len(data) > window:
                smoothed = smooth_data(data, window)
                ax.plot(smoothed, label='Shield', linewidth=2)
        if self.observation_features.get('hw_fan_on'):
            data = self.observation_features['hw_fan_on']
            if len(data) > window:
                smoothed = smooth_data(data, window)
                ax.plot(smoothed, label='Fan', linewidth=2, color='cyan')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Control Value')
        ax.set_title('Hardware Controls')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 14. Water Efficiency Metrics
        ax = fig.add_subplot(gs[4, 1])
        if self.observation_features.get('hw_water_efficiency'):
            data = self.observation_features['hw_water_efficiency']
            if len(data) > window:
                smoothed = smooth_data(data, window)
                ax.plot(smoothed, label='Water Efficiency', linewidth=2, color='blue')
        if self.observation_features.get('hw_coverage_efficiency'):
            data = self.observation_features['hw_coverage_efficiency']
            if len(data) > window:
                smoothed = smooth_data(data, window)
                ax.plot(smoothed, label='Coverage Efficiency', linewidth=2, color='green')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Efficiency')
        ax.set_title('Water & Coverage Efficiency')
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
        
        # 15. Action Usage
        ax = fig.add_subplot(gs[4, 2])
        for action_name, values in self.action_history.items():
            if len(values) > window:
                smoothed = smooth_data(values, window)
                ax.plot(smoothed, label=action_name, linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Mean Action Value')
        ax.set_title('Action Usage Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Row 5: Peltier Modules & Additional Metrics
        # 16. Peltier Module States
        ax = fig.add_subplot(gs[5, 0])
        peltier_plotted = False
        for i in range(4):  # Support up to 4 Peltier modules
            peltier_key = f'peltier_{i}'
            if self.observation_features.get(peltier_key):
                data = self.observation_features[peltier_key]
                if len(data) > window:
                    smoothed = smooth_data(data, window)
                    # Power is already in [-1, 1] range from true_state (no denormalization needed)
                    # Power range: -1 (cooling) to +1 (heating)
                    ax.plot(smoothed, label=f'Peltier {i+1}', linewidth=2)
                    peltier_plotted = True
        if not peltier_plotted:
            ax.text(0.5, 0.5, 'No Peltier data', ha='center', va='center', transform=ax.transAxes)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Power Level')
        ax.set_title('Peltier Module States (-1=cooling, +1=heating)')
        ax.legend()
        ax.set_ylim([-1.1, 1.1])
        ax.grid(True, alpha=0.3)
        
        # Row 6: Final Metrics
        # 17. Death Rate
        ax = fig.add_subplot(gs[5, 1])
        if len(self.episode_deaths) > window:
            death_rate = smooth_data(self.episode_deaths, window)
            ax.plot(death_rate * 100, linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Death Rate (%)')
        ax.set_title(f'Plant Death Rate (window={window})')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3)
        
        # 18. Water Use Efficiency
        ax = fig.add_subplot(gs[5, 2])
        efficiency = []
        for g, w in zip(self.episode_growth, self.episode_water_use):
            if w > 0:
                efficiency.append(g / w)
            else:
                efficiency.append(0)
        if len(efficiency) > window:
            smoothed = smooth_data(efficiency, window)
            ax.plot(smoothed, linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Growth / Water (g/L)')
        ax.set_title('Water Use Efficiency')
        ax.grid(True, alpha=0.3)
        
        # FIXED: Log termination reasons and time-to-failure
        # This is handled in log_episode via death_reason
        
        # 19. Plant Nutrient & NSC
        ax = fig.add_subplot(gs[6, 0])
        ax2_nsc = None
        if self.observation_features.get('plant_nutrient'):
            data = self.observation_features['plant_nutrient']
            if len(data) > window:
                smoothed = smooth_data(data, window)
                ax.plot(smoothed, label='Nutrient', linewidth=2, color='brown')
        if self.observation_features.get('plant_NSC'):
            data = self.observation_features['plant_NSC']
            if len(data) > window:
                smoothed = smooth_data(data, window)
                # Denormalize: NSC/5, so NSC = smoothed*5
                nsc_value = smoothed * 5
                ax2_nsc = ax.twinx()
                ax2_nsc.plot(nsc_value, label='NSC', color='green', linewidth=2)
                ax2_nsc.set_ylabel('NSC (g)', color='green')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Nutrient Level')
        ax.set_title('Plant Nutrient & NSC')
        ax.legend(loc='upper left')
        if ax2_nsc is not None:
            ax2_nsc.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.savefig(self.log_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved comprehensive training progress plot to {self.log_dir / 'training_progress.png'}")
    
    def plot_episode_detail(self, episode_data: List[Dict[str, Any]]):
        """Plot detailed view of a single episode"""
        fig, axes = plt.subplots(4, 2, figsize=(15, 12))
        fig.suptitle('Episode Detail', fontsize=16)
        
        # Extract time series
        hours = list(range(len(episode_data)))
        biomass = [d['plant']['biomass_total'] for d in episode_data]
        moisture = [d['plant']['soil_moisture'] for d in episode_data]
        LAI = [d['plant']['LAI'] for d in episode_data]
        temp = [d['env']['T'] for d in episode_data]
        light = [d['env']['L'] for d in episode_data]
        
        actions_water = [d['action'][0] for d in episode_data]
        actions_fan = [d['action'][1] for d in episode_data]
        actions_heater = [d['action'][3] for d in episode_data]
        
        rewards = [d['reward'] for d in episode_data]
        stress_water = [d['plant']['stress_water'] for d in episode_data]
        stress_temp = [d['plant']['stress_temp'] for d in episode_data]
        
        # 1. Biomass Growth
        ax = axes[0, 0]
        ax.plot(hours, biomass)
        ax.set_xlabel('Hour')
        ax.set_ylabel('Biomass (g)')
        ax.set_title('Total Biomass')
        ax.grid(True, alpha=0.3)
        
        # 2. LAI
        ax = axes[0, 1]
        ax.plot(hours, LAI, color='green')
        ax.set_xlabel('Hour')
        ax.set_ylabel('LAI')
        ax.set_title('Leaf Area Index')
        ax.grid(True, alpha=0.3)
        
        # 3. Soil Moisture + Water Action
        ax = axes[1, 0]
        ax.plot(hours, moisture, label='Moisture', color='blue')
        ax.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Optimal')
        ax.axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='Stress')
        ax2 = ax.twinx()
        ax2.bar(hours, actions_water, alpha=0.3, color='cyan', label='Water Action')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Soil Moisture')
        ax2.set_ylabel('Water Action')
        ax.set_title('Moisture Management')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 4. Temperature + Climate Control
        ax = axes[1, 1]
        ax.plot(hours, temp, label='Temperature', color='red')
        ax.axhline(y=22, color='g', linestyle='--', alpha=0.5, label='Optimal')
        ax2 = ax.twinx()
        ax2.plot(hours, actions_fan, alpha=0.5, color='blue', label='Fan')
        ax2.plot(hours, actions_heater, alpha=0.5, color='orange', label='Heater')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Temperature (°C)')
        ax2.set_ylabel('Action Value')
        ax.set_title('Climate Control')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 5. Light Environment
        ax = axes[2, 0]
        ax.plot(hours, light, color='gold')
        ax.fill_between(hours, 0, light, alpha=0.3, color='yellow')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Light (normalized)')
        ax.set_title('Light Availability')
        ax.grid(True, alpha=0.3)
        
        # 6. Reward Signal
        ax = axes[2, 1]
        ax.plot(hours, rewards)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Hour')
        ax.set_ylabel('Reward')
        ax.set_title('Reward per Step')
        ax.grid(True, alpha=0.3)
        
        # 7. Stress Factors
        ax = axes[3, 0]
        ax.plot(hours, stress_water, label='Water Stress', color='blue')
        ax.plot(hours, stress_temp, label='Temp Stress', color='red')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Threshold')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Stress Factor')
        ax.set_title('Plant Stress Levels')
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
        
        # 8. Action Summary
        ax = axes[3, 1]
        action_means = {
            'Water': np.mean(actions_water),
            'Fan': np.mean(actions_fan),
            'Heater': np.mean(actions_heater)
        }
        ax.bar(action_means.keys(), action_means.values())
        ax.set_ylabel('Mean Action Value')
        ax.set_title('Action Usage Summary')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'episode_detail.png', dpi=150)
        plt.close()
        
        print(f"✓ Saved episode detail plot to {self.log_dir / 'episode_detail.png'}")
    
    def save_summary(self):
        """Save training summary statistics"""
        summary = {
            'total_episodes': len(self.episode_rewards),
            'mean_reward': float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0,
            'mean_growth': float(np.mean(self.episode_growth[-100:])) if self.episode_growth else 0,
            'mean_survival': float(np.mean(self.episode_lengths[-100:])) if self.episode_lengths else 0,
            'death_rate': float(np.mean(self.episode_deaths[-100:])) if self.episode_deaths else 0,
            'mean_water_efficiency': float(np.mean([
                g/w for g, w in zip(self.episode_growth[-100:], self.episode_water_use[-100:]) 
                if w > 0
            ])) if self.episode_growth else 0,
        }
        
        with open(self.log_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY (last 100 episodes)")
        print("="*60)
        print(f"Mean Reward:          {summary['mean_reward']:.2f}")
        print(f"Mean Growth:          {summary['mean_growth']:.2f} g")
        print(f"Mean Survival:        {summary['mean_survival']:.1f} / 336 steps")
        print(f"Death Rate:           {summary['death_rate']*100:.1f}%")
        print(f"Water Efficiency:     {summary['mean_water_efficiency']:.2f} g/L")
        print("="*60)


def run_episode_with_logging(env, model, deterministic=True) -> List[Dict[str, Any]]:
    """Run a single episode and collect detailed data"""
    episode_data = []
    
    obs, info = env.reset()
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_data.append({
            'obs': obs.copy(),
            'action': action.copy(),
            'reward': float(reward),
            'plant': info['plant'].copy(),
            'env': info['env'].copy(),
            'hw': info['hw'].copy(),
        })
        
        done = terminated or truncated
    
    return episode_data


# Example usage in training script:
"""
from rl.diagnostics import TrainingDiagnostics

diagnostics = TrainingDiagnostics()

# During training (in callback or after episodes)
for episode in range(num_episodes):
    # ... run episode ...
    diagnostics.log_episode(info, actions_taken)

# Periodically (every 100 episodes)
if episode % 100 == 0:
    diagnostics.plot_training_progress()
    diagnostics.save_summary()

# After training completes
diagnostics.plot_training_progress()
diagnostics.save_summary()

# Analyze a specific episode
episode_data = run_episode_with_logging(env, model)
diagnostics.plot_episode_detail(episode_data)
"""