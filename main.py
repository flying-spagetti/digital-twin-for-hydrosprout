#!/usr/bin/env python3
"""
main.py - Orchestrator for the Digital Twin project

Usage examples:
    python main.py gen_synth --out synth_images --n 300
    python main.py demo_ppo
    python main.py train_classifier --data synth_images --epochs 5
    python main.py sim_run --steps 48
    python main.py sample_visual --out sample.png

This script expects to be run from the project root (digital-twin/).
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Local modules (from template)
from visionmodel import synth_generator
from rl.train_ppo import main as ppo_main
from visionmodel import train_classifier
from rl.gym_env import DigitalTwinEnv

# Default uploaded image path from session (you can change this)
DEFAULT_REAL_IMAGE = "/mnt/data/A_high-resolution_digital_photograph_showcases_an_.png"

def gen_synth(args):
    out = args.out or "synth_images"
    n = args.n or 200
    print(f"[main] Generating {n} synthetic images into {out} ...")
    synth_generator.gen_dataset(out_dir=out, n=n)
    print("[main] Done.")

def demo_ppo(args):
    print("[main] Running PPO demo training (quick)...")
    # reuse rl/train_ppo.py entrypoint with --demo
    # Note: the train_ppo.main expects parsed args; we call it via subprocess-like interface
    class Args: demo = True
    ppo_main(Args())
    print("[main] PPO demo finished. Model saved as ppo_demo.")

def train_classifier_cmd(args):
    data_dir = args.data or "synth_images"
    epochs = args.epochs or 5
    print(f"[main] Training classifier on '{data_dir}' for {epochs} epochs ...")
    train_classifier.train(data_dir=data_dir, epochs=epochs)
    print("[main] Classifier training done.")

def sim_run(args):
    steps = args.steps or 48
    env = DigitalTwinEnv()
    obs,info=env.reset()
    print("[main] Starting sim run for", steps, "steps")
    total_reward = 0.0
    for t in range(steps):
        # naive policy: water if moisture low, toggle fan by temp, tiny shield adjustments
        canopy, moist, nut, pmold, tmp_scaled, lux, shield_pos = obs
        temp = tmp_scaled * 40.0
        action = [0.0, 0.0, 0.0, 0.0]  # water, fan, shield_delta, heater
        if moist < 0.35:
            action[0] = 0.8  # water
        if temp > 28.0:
            action[1] = 1.0  # fan on
            action[2] = 0.2  # open shield slightly
        if temp < 16.0:
            action[3] = 0.6  # heater
        obs, rew, terminated, truncated, info = env.step(action)
        total_reward += rew
        if t % 6 == 0:
            env.render()
        if terminated or truncated:
            break
    print("[main] Sim finished. Total reward:", total_reward)

def sample_visual(args):
    out = args.out or "sample_visual.png"
    # create a single synthetic image and optionally composite with provided real image
    tmp_dir = ROOT / "tmp_sample"
    tmp_dir.mkdir(exist_ok=True)
    synth_path = tmp_dir / "synth_sample.png"
    print("[main] Generating one synthetic sample...")
    synth_generator.gen_dataset(out_dir=str(tmp_dir), n=1)
    synth_img = tmp_dir / "synth_0000.png"
    # if a real image exists, create a simple composite
    real_path = args.real or DEFAULT_REAL_IMAGE
    if os.path.exists(real_path):
        try:
            from PIL import Image
            base = Image.open(real_path).convert("RGBA").resize((224,224))
            overlay = Image.open(str(synth_img)).convert("RGBA").resize((224,224))
            # composite: overlay canopy with 50% alpha
            blended = Image.blend(base, overlay, alpha=0.55)
            blended.save(out)
            print(f"[main] Composite saved to {out} (using real image {real_path})")
            return
        except Exception as e:
            print("[main] Composite failed:", e)
    # otherwise just copy the synthetic image
    os.replace(str(synth_img), out)
    print(f"[main] Sample synthetic image saved to {out}")

def parse_args():
    p = argparse.ArgumentParser(description="Digital Twin - main orchestrator")
    sub = p.add_subparsers(dest="cmd")

    g = sub.add_parser("gen_synth", help="Generate synthetic images")
    g.add_argument("--out", type=str, help="output directory")
    g.add_argument("--n", type=int, help="number of images")

    d = sub.add_parser("demo_ppo", help="Run quick PPO demo training")

    tc = sub.add_parser("train_classifier", help="Train classifier on dataset")
    tc.add_argument("--data", type=str, help="data directory (ImageFolder layout)")
    tc.add_argument("--epochs", type=int, help="epochs")

    s = sub.add_parser("sim_run", help="Run a short simulation with heuristic policy")
    s.add_argument("--steps", type=int, help="timesteps to run")

    sv = sub.add_parser("sample_visual", help="Generate sample visual and optionally composite with real image")
    sv.add_argument("--out", type=str, help="output filename")
    sv.add_argument("--real", type=str, help="path to real image to composite with")

    return p.parse_args()

def main():
    args = parse_args()
    if not args.cmd:
        print("No command given. Use -h to see options.")
        return
    if args.cmd == "gen_synth":
        gen_synth(args)
    elif args.cmd == "demo_ppo":
        demo_ppo(args)
    elif args.cmd == "train_classifier":
        train_classifier_cmd(args)
    elif args.cmd == "sim_run":
        sim_run(args)
    elif args.cmd == "sample_visual":
        sample_visual(args)
    else:
        print("Unknown command:", args.cmd)

if __name__ == "__main__":
    main()
