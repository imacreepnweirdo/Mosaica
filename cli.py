"""
Usage:
    python -m mosaica.cli --mode color --target in.jpg --output out.png --tile-width 10 --tile-height 10
    python -m mosaica.cli --mode photo --target in.jpg --output out.png --codebook-dir photos/ --tile-width 16 --tile-height 16
"""

import argparse

from mosaica.core.config import MosaicConfig, build_renderer
from mosaica.core.pipeline import run_mosaic


def main():
    parser = argparse.ArgumentParser(description="Generate a mosaic from a target image.")

    parser.add_argument("--mode", choices=["color", "photo"], required=True)
    parser.add_argument("--target", required=True, help="Path to the target image")
    parser.add_argument("--output", required=True, help="Where to save the mosaic")
    parser.add_argument("--tile-width", type=int, default=16)
    parser.add_argument("--tile-height", type=int, default=16)
    parser.add_argument("--enlargement", type=int, default=1, help="Upscale target before tiling for more detail")

    # photo mode only
    parser.add_argument("--codebook-dir", default=None, help="Folder of images to use as tiles (photo mode)")
    parser.add_argument("--matcher", choices=["color", "embedding"], default="color",
                         help="How to match tiles: mean color (fast) or pretrained-CNN embeddings (slower, smarter)")
    parser.add_argument("--avoid-repeat-within", type=int, default=6)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--no-cache", action="store_true")

    args = parser.parse_args()

    config = MosaicConfig(
        mode=args.mode,
        target_path=args.target,
        output_path=args.output,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        enlargement=args.enlargement,
        codebook_dir=args.codebook_dir,
        matcher=args.matcher,
        avoid_repeat_within=args.avoid_repeat_within,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
    )

    renderer = build_renderer(config)
    run_mosaic(config.target_path, config.output_path, renderer, config.tile_width, config.tile_height, config.enlargement)


if __name__ == "__main__":
    main()
