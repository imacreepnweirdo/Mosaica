"""
Turns a folder of photos into a list of tile-sized PIL Images ("the codebook"),
with disk caching so repeated runs on the same folder are instant.
"""

import io
from PIL import Image

from mosaica.codebook import scanner, cache


def _resize_cover(image, width, height):
    """
    Resize + center-crop to exactly (width, height), preserving aspect ratio
    (like CSS 'object-fit: cover') instead of squishing the source image.
    """

    src_ratio = image.width / image.height
    dst_ratio = width / height

    if src_ratio > dst_ratio:
        new_height = height
        new_width = int(height * src_ratio)
    else:
        new_width = width
        new_height = int(width / src_ratio)

    resized = image.resize((new_width, new_height), Image.LANCZOS)

    left = (new_width - width) // 2
    top = (new_height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _to_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def load_codebook(codebook_dir, tile_width, tile_height, cache_dir="cache", use_cache=True):
    """
    Returns a list of PIL Images, one per usable photo in codebook_dir, each
    resized to (tile_width, tile_height).
    """

    signature = scanner.folder_signature(codebook_dir)
    cache_key = f"codebook_{signature}_{tile_width}x{tile_height}"

    if use_cache:
        cached = cache.load(cache_dir, cache_key)
        if cached is not None:
            print(f"[codebook] loaded {len(cached)} cached tiles")
            return [Image.open(io.BytesIO(b)) for b in cached]

    paths = scanner.find_images(codebook_dir)
    if not paths:
        raise ValueError(f"No usable images found in {codebook_dir}")

    tiles = []
    for path in paths:
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[codebook] skipping {path}: {e}")
            continue
        tiles.append(_resize_cover(image, tile_width, tile_height))

    print(f"[codebook] built {len(tiles)} tiles from {codebook_dir}")

    if use_cache:
        cache.save(cache_dir, cache_key, [_to_bytes(t) for t in tiles])

    return tiles
