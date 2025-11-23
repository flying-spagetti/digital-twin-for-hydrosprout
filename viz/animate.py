# viz/animate.py
"""
Simple animation utilities for the Digital Twin simulator.

Provides functions to:
- animate canopy growth frames into an MP4
- render a live simulation as a sequence of PNG frames

Requires imageio and matplotlib for making MP4s.
"""

import os
import imageio
import numpy as np
from PIL import Image, ImageDraw

DEFAULT_REAL_IMAGE = '/mnt/data/A_high-resolution_digital_photograph_showcases_an_.png'


def render_canopy_frame(canopy_fraction, width=512, height=384, base_color=(40,160,40)):
    """Render a simple canopy-like image for a given canopy fraction (0..1)."""
    # generate a mask density proportional to canopy_fraction
    from vision.synth_generator import generate_canopy_mask, colorize_mask
    mask = generate_canopy_mask(width=width, height=height, density=max(0.05, canopy_fraction*0.9))
    img = colorize_mask(mask, base_color=base_color)
    return img


def render_sequence(frames, out_dir='anim_frames'):
    """Save a list of PIL images to out_dir as numbered PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, img in enumerate(frames):
        p = os.path.join(out_dir, f'frame_{i:04d}.png')
        img.save(p)
        paths.append(p)
    return paths


def frames_to_mp4(frame_paths, out_path='animation.mp4', fps=6):
    """Stitch frames into an MP4 using imageio."""
    writer = imageio.get_writer(out_path, fps=fps)
    for p in frame_paths:
        img = imageio.imread(p)
        writer.append_data(img)
    writer.close()
    print(f'[viz] Wrote animation to {out_path}')


def demo_animation(out_path='demo_anim.mp4', steps=48):
    """Create a demo animation by simulating simple canopy growth curve."""
    frames = []
    for t in range(steps):
        # simple logistic-like growth for demo
        frac = 1.0 / (1.0 + np.exp(-0.08*(t-steps/2)))
        img = render_canopy_frame(frac)
        frames.append(img)
    paths = render_sequence(frames, out_dir='anim_demo_frames')
    frames_to_mp4(paths, out_path=out_path, fps=6)
    return out_path


def composite_with_real(base_real_path=DEFAULT_REAL_IMAGE, overlay_img=None, out_path='composite.png'):
    """Composite an overlay (PIL.Image) onto a real photo and save."""
    try:
        base = Image.open(base_real_path).convert('RGBA').resize((224,224))
    except Exception as e:
        raise RuntimeError(f'Failed to open base image at {base_real_path}: {e}')
    if overlay_img is None:
        # generate a single sample overlay
        overlay_img = render_canopy_frame(0.6).resize((224,224)).convert('RGBA')
    blended = Image.blend(base, overlay_img.convert('RGBA'), alpha=0.55)
    blended.save(out_path)
    print(f'[viz] Saved composite to {out_path}')
    return out_path
