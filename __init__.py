"""
mosaica — a small, extensible mosaic-generation engine.

Two rendering modes share one pipeline:
  - "color": each grid cell is redrawn as a flat block of its average color
             (painterly / pixel-art style, no codebook needed).
  - "photo": each grid cell is replaced by the closest-matching real image
             from a folder of your own photos (a true photomosaic).
"""

__version__ = "0.1.0"
