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
        
        # Plant state history
        self.plant_states = {
            'biomass': [],
            'moisture': [],
            'LAI': [],
            'temp': [],
        }
    
    def log_episode(self, info: Dict[str, Any], actions: List[np.ndarray]):
        """Log episode statistics"""
        self.episode_rewards.append(info.get('episode_reward', 0))
        self.episode_lengths.append(info.get('episode_length', 0))
        self.episode_growth.append(info.get('cumulative_growth', 0))
        self.episode_deaths.append(1 if info.get('death_reason') else 0)
        self.episode_water_use.append(info.get('cumulative_water_used', 0))
        
        # Aggregate action statistics
        if actions:
            actions_array = np.array(actions)
            self.action_history['water'].append(actions_array[:, 0].mean())
            self.action_history['fan'].append(actions_array[:, 1].mean())
            self.action_history['shield'].append(actions_array[:, 2].mean())
            self.action_history['heater'].append(actions_array[:, 3].mean())
    
    def log_step(self, plant_state: Dict[str, Any], env_state: Dict[str, Any]):
        """Log individual step for detailed analysis"""
        self.plant_states['biomass'].append(plant_state.get('biomass_total', 0))
        self.plant_states['moisture'].append(plant_state.get('soil_moisture', 0))
        self.plant_states['LAI'].append(plant_state.get('LAI', 0))
        self.plant_states['temp'].append(env_state.get('T', 20))
    
    def plot_training_progress(self, window: int = 100):
        """Plot comprehensive training metrics"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Training Progress', fontsize=16)
        
        # 1. Episode Rewards (smoothed)
        ax = axes[0, 0]
        if len(self.episode_rewards) > window:
            smoothed = np.convolve(self.episode_rewards, 
                                  np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=f'Smoothed (window={window})')
        ax.plot(self.episode_rewards, alpha=0.3, label='Raw')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Episode Lengths (survival time)
        ax = axes[0, 1]
        ax.plot(self.episode_lengths)
        ax.axhline(y=336, color='r', linestyle='--', label='Max (14 days)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Length (Survival)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Cumulative Growth
        ax = axes[1, 0]
        if len(self.episode_growth) > window:
            smoothed = np.convolve(self.episode_growth, 
                                  np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=f'Smoothed')
        ax.plot(self.episode_growth, alpha=0.3, label='Raw')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Growth (g)')
        ax.set_title('Cumulative Biomass Gain')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Death Rate
        ax = axes[1, 1]
        if len(self.episode_deaths) > window:
            death_rate = np.convolve(self.episode_deaths, 
                                    np.ones(window)/window, mode='valid')
            ax.plot(death_rate * 100)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Death Rate (%)')
        ax.set_title(f'Plant Death Rate (window={window})')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3)
        
        # 5. Action Usage
        ax = axes[2, 0]
        for action_name, values in self.action_history.items():
            if len(values) > window:
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                ax.plot(smoothed, label=action_name)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Mean Action Value')
        ax.set_title('Action Usage Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Water Use Efficiency
        ax = axes[2, 1]
        efficiency = []
        for g, w in zip(self.episode_growth, self.episode_water_use):
            if w > 0:
                efficiency.append(g / w)
            else:
                efficiency.append(0)
        
        if len(efficiency) > window:
            smoothed = np.convolve(efficiency, np.ones(window)/window, mode='valid')
            ax.plot(smoothed)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Growth / Water (g/L)')
        ax.set_title('Water Use Efficiency')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_progress.png', dpi=150)
        plt.close()
        
        print(f"✓ Saved training progress plot to {self.log_dir / 'training_progress.png'}")
    
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