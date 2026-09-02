"""
Color-block rendering — no codebook needed. Each cell becomes a flat square
of its own average color. This is your original pixel_mosaic.py, expressed
as a Renderer.
"""

import numpy as np
from PIL import Image
from mosaica.rendering.base import Renderer


class ColorBlockRenderer(Renderer):
    def __init__(self, tile_width, tile_height, gap=0):
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.gap = gap

    def render(self, region_image):
        avg_color = tuple(
            int(c) for c in np.array(region_image.getdata()).mean(axis=0)
        )

        patch = Image.new("RGB", (self.tile_width, self.tile_height), avg_color)

        if self.gap > 0:
            # shrink the drawn square within its cell to leave a visible gap,
            # matching the "gap" look from the original script
            bg = Image.new("RGB", (self.tile_width, self.tile_height), "white")
            inner = patch.resize((self.tile_width - 2 * self.gap, self.tile_height - 2 * self.gap))
            bg.paste(inner, (self.gap, self.gap))
            return bg

        return patch
