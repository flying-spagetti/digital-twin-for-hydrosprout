# vision/synth_generator.py
"""
Synthetic Image Generator
------------------------
This module creates simple synthetic microgreens images by generating canopy-like masks,
colorizing them, and applying dust/noise overlays.

Run directly to generate a dataset:
    python vision/synth_generator.py
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------
def generate_canopy_mask(width=512, height=384, density=0.5):
    """
    Generates a grayscale mask where white regions represent leaf canopy.
    density: controls number of leaf blobs.
    """
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    num_blobs = int(40 * density)
    for _ in range(num_blobs):
        x = random.randint(0, width)
        y = random.randint(0, height)
        r = random.randint(10, 60)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=255)

    mask = mask.filter(ImageFilter.GaussianBlur(radius=5))
    return mask

# ---------------------------------------------------------------
def colorize_mask(mask, base_color=(40, 160, 40)):
    """
    Colors the canopy mask with a green tone over a neutral background.
    """
    base = Image.new('RGB', mask.size, (200, 200, 200))
    green = Image.new('RGB', mask.size, base_color)
    return Image.composite(green, base, mask)

# ---------------------------------------------------------------
def add_dust(img):
    """
    Adds noise simulating environmental dust.
    """
    w, h = img.size
    dust = Image.effect_noise((w, h), 64).convert('L')
    dust = dust.point(lambda p: p > 240 and 255)
    dust = dust.filter(ImageFilter.GaussianBlur(2))
    img.paste((180, 160, 160), (0, 0), dust)
    return img

# ---------------------------------------------------------------
def gen_dataset(out_dir='synth_images', n=200):
    """
    Generates n synthetic canopy images and stores them in out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    for i in range(n):
        density = random.uniform(0.1, 0.9)
        mask = generate_canopy_mask(density=density)
        img = colorize_mask(mask)

        if random.random() < 0.3:
            img = add_dust(img)

        img = img.resize((224, 224))
        img.save(os.path.join(out_dir, f'synth_{i:04d}.png'))

    print(f"Generated {n} synthetic images in {out_dir}")

# ---------------------------------------------------------------
if __name__ == '__main__':
    gen_dataset()
