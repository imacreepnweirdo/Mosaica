"""
Quick end-to-end smoke test — not a full test suite, just enough to catch
"I broke the pipeline" while we iterate. Run directly:

    PYTHONPATH=. python3 mosaica/tests/test_smoke.py
"""

import os
import random
import shutil
import tempfile

import numpy as np
from PIL import Image

from mosaica.core.config import MosaicConfig, build_renderer
from mosaica.core.pipeline import run_mosaic


def make_fixtures(tmp_dir):
    codebook_dir = os.path.join(tmp_dir, "codebook")
    os.makedirs(codebook_dir, exist_ok=True)

    random.seed(0)
    for i in range(20):
        color = tuple(random.randint(0, 255) for _ in range(3))
        Image.new("RGB", (32, 32), color).save(os.path.join(codebook_dir, f"t{i}.png"))

    target_path = os.path.join(tmp_dir, "target.png")
    target = Image.new("RGB", (100, 80))
    px = target.load()
    for x in range(100):
        for y in range(80):
            px[x, y] = (int(255 * x / 100), int(255 * y / 80), 128)
    target.save(target_path)

    return codebook_dir, target_path


def test_color_mode(tmp_dir, target_path):
    output_path = os.path.join(tmp_dir, "out_color.png")
    config = MosaicConfig(mode="color", target_path=target_path, output_path=output_path,
                           tile_width=10, tile_height=10)
    renderer = build_renderer(config)
    run_mosaic(target_path, output_path, renderer, 10, 10)

    assert os.path.exists(output_path)
    out = Image.open(output_path)
    assert out.size == (100, 80)
    print("test_color_mode: OK")


def test_photo_mode(tmp_dir, target_path, codebook_dir):
    output_path = os.path.join(tmp_dir, "out_photo.png")
    cache_dir = os.path.join(tmp_dir, "cache")
    config = MosaicConfig(mode="photo", target_path=target_path, output_path=output_path,
                           tile_width=8, tile_height=8, codebook_dir=codebook_dir,
                           avoid_repeat_within=4, cache_dir=cache_dir)
    renderer = build_renderer(config)
    run_mosaic(target_path, output_path, renderer, 8, 8)

    assert os.path.exists(output_path)
    out = Image.open(output_path)
    assert out.width % 8 == 0 and out.height % 8 == 0
    print("test_photo_mode: OK")

    # second run should hit the cache without error
    renderer2 = build_renderer(config)
    run_mosaic(target_path, os.path.join(tmp_dir, "out_photo2.png"), renderer2, 8, 8)
    print("test_photo_mode (cached): OK")


def test_pretrained_embedding_shape(tmp_dir):
    from mosaica.models.pretrained_embedding import PretrainedEmbeddingModel

    target_path = os.path.join(tmp_dir, "embedding_target.png")
    img = Image.new("RGB", (64, 64), (12, 34, 56))
    img.save(target_path)

    model = PretrainedEmbeddingModel()
    embedding = model.embed_image(Image.open(target_path).convert("RGB"))

    assert embedding.shape == (512,)
    assert np.isfinite(embedding).all()
    print("test_pretrained_embedding_shape: OK")


def test_invalid_mode_raises(tmp_dir, target_path):
    try:
        build_renderer(MosaicConfig(mode="bogus", target_path=target_path, output_path="x.png"))
    except ValueError:
        print("test_invalid_mode_raises: OK")
        return
    raise AssertionError("expected ValueError for unknown mode")


if __name__ == "__main__":
    tmp_dir = tempfile.mkdtemp()
    try:
        codebook_dir, target_path = make_fixtures(tmp_dir)
        test_color_mode(tmp_dir, target_path)
        test_photo_mode(tmp_dir, target_path, codebook_dir)
        test_pretrained_embedding_shape(tmp_dir)
        test_invalid_mode_raises(tmp_dir, target_path)
        print("\nAll smoke tests passed.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
