# viz/plot_utils.py
"""
Utility plotting functions for the Digital Twin project.

Provides:
- time-series plotting of plant/env/hardware traces
- summary plots for episodes (canopy vs time, moisture, temp, reward)

Note: uses matplotlib and expects numeric data in sequences or pandas DataFrame-like dicts.
"""

import matplotlib.pyplot as plt
import os


def plot_time_series(log, out_path=None, title=None):
    """
    Plot basic time-series from a log dictionary.

    log: dict of lists, e.g. {
        'time': [0,1,2,...],
        'canopy': [...],
        'moisture': [...],
        'temp': [...],
        'lux': [...],
        'reward': [...]
    }
    """
    time = log.get('time', list(range(len(log.get('canopy', [])))))
    canopy = log.get('canopy', [])
    moisture = log.get('moisture', [])
    temp = log.get('temp', [])
    lux = log.get('lux', [])
    reward = log.get('reward', [])

    plt.figure(figsize=(12, 8))
    ax1 = plt.gca()
    ax1.plot(time, canopy, label='Canopy', linewidth=2)
    ax1.plot(time, moisture, label='Moisture')
    ax1.set_xlabel('Timestep (hours)')
    ax1.set_ylabel('Canopy / Moisture')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    if temp:
        ax2.plot(time, temp, label='Temp (°C)', color='tab:red', linestyle='--')
    if lux:
        ax2.plot(time, lux, label='Lux', color='tab:orange', linestyle=':')
    ax2.set_ylabel('Temp / Lux')
    ax2.legend(loc='upper right')

    if reward:
        plt.figure(figsize=(8,3))
        plt.plot(time, reward, label='Reward')
        plt.xlabel('Timestep (hours)')
        plt.ylabel('Reward')
        plt.title('Episode reward')
        if out_path:
            base, ext = os.path.splitext(out_path)
            rpath = base + '_reward' + (ext or '.png')
            plt.savefig(rpath, dpi=150)
        else:
            plt.show()

    if title:
        plt.suptitle(title)

    if out_path:
        plt.savefig(out_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_episode_summary(history, out_dir='plots', prefix='episode'):
    """Create and save a set of plots summarizing a single episode"""
    os.makedirs(out_dir, exist_ok=True)
    t = history.get('time', list(range(len(history.get('canopy', [])))))
    plot_time_series(history, out_path=os.path.join(out_dir, f'{prefix}_timeseries.png'), title=f'{prefix} summary')

    # additional quick stats
    canopy = history.get('canopy', [])
    if canopy:
        final = canopy[-1]
        best = max(canopy)
        with open(os.path.join(out_dir, f'{prefix}_summary.txt'), 'w') as f:
            f.write(f'final_canopy: {final}\n')
            f.write(f'peak_canopy: {best}\n')

    print(f'[viz] saved episode summary to {out_dir}/{prefix}_*')
