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
    matcher: str = "color"  # "color" or "embedding" — which FeatureExtractor to match tiles with


def build_renderer(config: MosaicConfig):
    if config.mode == "color":
        from mosaica.rendering.color_block import ColorBlockRenderer
        return ColorBlockRenderer(config.tile_width, config.tile_height)

    if config.mode == "photo":
        if not config.codebook_dir:
            raise ValueError("photo mode requires codebook_dir")

        from mosaica.codebook.loader import load_codebook
        from mosaica.codebook.scanner import folder_signature
        from mosaica.rendering.photo_tile import PhotoTileRenderer

        tiles = load_codebook(
            config.codebook_dir,
            config.tile_width,
            config.tile_height,
            cache_dir=config.cache_dir,
            use_cache=config.use_cache,
        )

        # folder_signature() changes automatically if the codebook folder's
        # contents change, so a stale feature cache never gets reused by
        # mistake — same content-hash approach load_codebook() uses for tiles.
        signature = folder_signature(config.codebook_dir)

        if config.matcher == "color":
            from mosaica.matching.color_matcher import build_color_matcher
            matcher = build_color_matcher(avoid_repeat_within=config.avoid_repeat_within)
            cache_key = None  # mean color is cheap enough that caching isn't worth it

        elif config.matcher == "embedding":
            from mosaica.matching.embedding_matcher import build_embedding_matcher
            matcher = build_embedding_matcher(avoid_repeat_within=config.avoid_repeat_within)
            cache_key = f"embeddings_{signature}_{config.tile_width}x{config.tile_height}_resnet18"

        else:
            raise ValueError(f"Unknown matcher: {config.matcher!r} (expected 'color' or 'embedding')")

        matcher.fit(
            tiles,
            cache_dir=config.cache_dir if (config.use_cache and cache_key) else None,
            cache_key=cache_key,
        )

        return PhotoTileRenderer(matcher, tiles)

    raise ValueError(f"Unknown mode: {config.mode!r} (expected 'color' or 'photo')")
