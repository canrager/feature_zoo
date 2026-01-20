"""
Docstring for exp.notes

Source: https://github.com/alexandermodell/Representation-Manifolds/blob/main/3-reproduce_figures.ipynb
Alexander Modell
"""

import pandas as pd
import numpy as np

from utils import *

def rgb_to_hls(r, g, b):
    """
    Convert RGB color values to HLS (Hue, Lightness, Saturation).

    Args:
        r, g, b: Red, Green, Blue values in range [0, 1]

    Returns:
        (h, l, s): Hue [0, 1], Lightness [0, 1], Saturation [0, 1]
    """
    maxc = max(r, g, b)
    minc = min(r, g, b)

    # Lightness
    l = (maxc + minc) / 2.0

    if maxc == minc:
        # Achromatic (gray)
        return 0.0, l, 0.0

    # Saturation
    if l <= 0.5:
        s = (maxc - minc) / (maxc + minc)
    else:
        s = (maxc - minc) / (2.0 - maxc - minc)

    # Hue
    delta = maxc - minc
    if r == maxc:
        h = (g - b) / delta
    elif g == maxc:
        h = 2.0 + (b - r) / delta
    else:  # b == maxc
        h = 4.0 + (r - g) / delta

    h = h / 6.0
    if h < 0.0:
        h += 1.0

    return h, l, s

def rgb_to_hex(rgb):
    r, g, b = rgb
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def hex_to_rgb(hex_codes):
    # Remove the '#' if it exists and convert hex codes to RGB values
    rgb_values = np.array([tuple(int(hex_code.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) for hex_code in hex_codes])
    return rgb_values

def hex_to_hls(hex_codes):
    # Convert hex codes to RGB values
    rgb_values = hex_to_rgb(hex_codes)
    # Normalize RGB values to the range [0, 1]
    rgb_normalized = rgb_values / 255.0
    # Convert RGB to HLS
    hls_values = np.array([rgb_to_hls(r, g, b) for r, g, b in rgb_normalized])
    return hls_values

# English names for colours
# get colour names and hex codes from XKCD survey
f = "https://xkcd.com/color/rgb.txt"
df = pd.read_csv(f, skiprows=1, delimiter="\t", names=["name", "hex"], index_col=0, usecols=[0,1])
colors = np.array(df.index)
hex = np.array(df.hex)
hls_vals = hex_to_hls(df['hex'])
hue = hls_vals[:,0]
lightness = hls_vals[:,1]
saturation = hls_vals[:,2]
# read in the embeddings (already normalised)

filter_words = [
    'neon', 'easter', 'iris', 'aqua', 'ugly', 'barf', 'grapefruit', 'apple', 'pear', 'melon', 'pea',
    'dirty', 'puke', 'vomit', 'barf', 'shit', 'poop', 'booger', 'snot', 'strawberry', 'topaz', 
    'sapphire', 'jade', 'poo', 'barney', 'cornflower', 'berry', 'coral', 'heliotrope', 'adobe',
    'hazel', 'merlot', 'bordeaux', 'wine', 'cheese', 'diarrhea', 'petrol', 'spruce', 'hot',
    'electric', 'royal', 'velvet'
]

# filter out low saturation, high lightness and colours which do not obviously refer to a colour.
masks = []
masks.append(
    np.full(len(colors), True)
)
masks.append(
    (saturation >= 0.4)
)
masks.append(
    (lightness <= 0.8)
)
for word in filter_words:
    masks.append(
        np.array([word not in c for c in colors])
    )

mask = np.logical_and.reduce(masks)

colors = colors[mask]
hex = hex[mask]
hue = hue[mask]
saturation = saturation[mask]
lightness = lightness[mask]

with open("data/texts/colors.txt", "w") as f:
    f.write("\n".join(colors))