"""
One place to describe a mosaic job, and one function that turns that
description into a ready-to-use Renderer. This is what keeps cli.py (and any
future GUI/notebook usage) from needing to know how each mode is wired up.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MosaicConfig:
    mode: str  # "color" or "photo"
    target_path: str
    output_path: str
    tile_width: int = 16
    tile_height: int = 16
    enlargement: int = 1

    # photo mode only
    codebook_dir: Optional[str] = None
    avoid_repeat_within: int = 6
    cache_dir: str = "cache"
    use_cache: bool = True


def build_renderer(config: MosaicConfig):
    if config.mode == "color":
        from mosaica.rendering.color_block import ColorBlockRenderer
        return ColorBlockRenderer(config.tile_width, config.tile_height)

    if config.mode == "photo":
        if not config.codebook_dir:
            raise ValueError("photo mode requires codebook_dir")

        from mosaica.codebook.loader import load_codebook
        from mosaica.matching.color_matcher import build_color_matcher
        from mosaica.rendering.photo_tile import PhotoTileRenderer

        tiles = load_codebook(
            config.codebook_dir,
            config.tile_width,
            config.tile_height,
            cache_dir=config.cache_dir,
            use_cache=config.use_cache,
        )

        matcher = build_color_matcher(avoid_repeat_within=config.avoid_repeat_within)
        matcher.fit(tiles)

        return PhotoTileRenderer(matcher, tiles)

    raise ValueError(f"Unknown mode: {config.mode!r} (expected 'color' or 'photo')")
