# mosaica

A mosaic-generation engine, built as a learning project while picking up PyTorch and CNNs.

Two rendering modes share one pipeline:

- **`color` mode** — each grid cell is redrawn as a flat square of its own average color (a painterly / pixel-art effect). No codebook needed.
- **`photo` mode** — each grid cell is replaced by the closest-matching real photo from a folder you provide (a true photomosaic).

Inspired by [worldveil/photomosaic](https://github.com/worldveil/photomosaic), but built independently and diverges from it in a few deliberate ways — see [Design notes](#design-notes-vs-the-reference-project) below.

## Why this project exists

It started as a simple "average the color of a region, draw a square" script. The goal became to extend that into two directions at once:

1. A real photomosaic — matching regions of a target image to real photos, not just flat color.
2. A vehicle for learning PyTorch/CNNs hands-on, by using a pretrained CNN to make that matching smarter than plain color-averaging.

## Architecture

The codebase separates three concerns on purpose, so that upgrading *how matching works* never requires touching the pipeline, the CLI, or how output actually gets rendered:

```
target image
     │
     ▼
  core/grid.py           splits the image into a grid of cells
     │
     ▼
  rendering/*.py          Renderer.render(region) → what to draw in this cell
     │            (color-block: draws avg color | photo-tile: pastes matched image)
     ▼
  matching/*.py            (photo mode only) — how the closest tile is found
     │
     ▼
  core/pipeline.py         assembles cells into the final output image
```

- **`FeatureExtractor`** (`matching/base.py`) — turns an image into a feature vector for similarity search. Two implementations exist:
  - `ColorFeatureExtractor` (`matching/color_matcher.py`) — mean RGB, 3 numbers.
  - `PretrainedEmbeddingModel` (`models/pretrained_embedding.py`) — a pretrained ResNet-18 with its classifier head removed, 512 numbers. *(Currently a standalone tool for inspecting embeddings — not yet wired into matching. That's Phase 3.)*
- **`AvoidanceStrategy`** (`matching/strategies.py`) — given a ranked list of candidate matches, decides which to actually use, so the same tile doesn't repeat endlessly across flat regions of the target. `CooldownAvoidance` skips any tile used within the last *N* picks.
- **`Renderer`** (`rendering/base.py`) — decides what actually gets drawn per cell. `ColorBlockRenderer` needs nothing but the cropped region. `PhotoTileRenderer` wraps a fitted matcher. The pipeline calls `renderer.render(region)` and has no idea which mode is running.
- **`NearestNeighborMatcher`** (`matching/base.py`) combines any `FeatureExtractor` + any `AvoidanceStrategy` — new matching techniques mean writing a new `FeatureExtractor`, not new pipeline logic.

### Project layout

```
mosaica/
  core/       grid.py (cell splitting), pipeline.py (orchestration), config.py (mode → Renderer factory)
  codebook/   scanner.py (find/fingerprint images), cache.py (generic pickle cache), loader.py (build tile codebook)
  matching/   base.py (interfaces), color_matcher.py, strategies.py
  rendering/  base.py (interface), color_block.py, photo_tile.py
  models/     pretrained_embedding.py — ResNet-18 feature extractor (Phase 2)
  cli.py      argparse entry point
  tests/      test_smoke.py — end-to-end check, no pytest dependency
```

## Status

**Phase 1 — Core architecture ✅**
Grid splitting, codebook loading with content-hash caching, color and photo rendering modes, cooldown-based repeat avoidance, CLI, smoke tests. Both modes tested end-to-end.

**Phase 2 — PyTorch introduction ✅**
`models/pretrained_embedding.py` wraps a pretrained ResNet-18 (`torchvision`, `ResNet18_Weights.DEFAULT`) with its final classification layer replaced by `nn.Identity()`, so it outputs a 512-dimensional feature vector instead of class predictions. Preprocessing follows the standard ImageNet pipeline: resize → center-crop to 224×224 → normalize with ImageNet mean/std. Inference runs in `.eval()` mode under `torch.no_grad()`.

This phase was spent understanding what these embeddings actually are, not just wiring them in:

- The 512 values are post-ReLU activations, so every value is ≥ 0 — a 0 means "this particular learned pattern-detector didn't fire for this image," not a missing/invalid value.
- Comparing raw stats (mean, max) between two embeddings is misleading — two very different images can coincidentally have similar means. **Cosine similarity** (comparing the *direction* the two vectors point in) is the correct comparison.
- Self-similarity sanity check: comparing an image's embedding against itself gives cosine similarity ≈ `1.0` (confirms the pipeline is correct).
- Real test: a bird photo vs. a tree photo scored `~0.53` cosine similarity — not near 1.0 (different subjects) but well above 0 (shared visual structure: foliage, natural lighting, organic textures) — the kind of nuance mean-RGB color alone could never capture.

**Phase 3 — Embedding-based matching (next)**
`matching/embedding_matcher.py`: an `EmbeddingFeatureExtractor` implementing the same `FeatureExtractor` interface as `ColorFeatureExtractor`, using cosine similarity (via L2-normalizing vectors, then reusing the existing `cKDTree` euclidean search — euclidean distance on unit-length vectors is monotonic with cosine similarity, so no new search infrastructure is needed). Plugs into `NearestNeighborMatcher` with zero changes to `pipeline.py`, `PhotoTileRenderer`, or `cli.py`.

**Phase 4 — Analysis tools (planned)**
`analysis/` — visualize a codebook's embedding space (t-SNE/PCA), and a metric quantifying how close a finished mosaic actually is to its target.

**Phase 5 — Stretch goal**
Fine-tune / metric-learn a small embedding model on your own codebook, instead of relying solely on a generic pretrained network.

## Design notes vs. the reference project

[worldveil/photomosaic](https://github.com/worldveil/photomosaic) does true photomosaics using `faiss` for nearest-neighbor search, but its matching is based on **flattened, resized pixel vectors** (still fundamentally raw color/position data, just higher-resolution than a single average), and near-duplicate logic is spread across several top-level scripts (`mosaic.py`, `video.py`, `interactive.py`, `make_gif.py`).

This project differs on purpose:
- One pipeline with pluggable Renderer/Matcher pieces, instead of parallel scripts per feature.
- Matching moves toward real CNN feature embeddings (Phase 3), which capture texture/shape/structure — not achievable with flattened pixel vectors, however high-resolution.
- Content-hash-based caching from the start (vs. path/filename-based caching, which breaks more easily).
- Two genuinely different mosaic aesthetics (flat color-block vs. photo-tile) sharing one engine, instead of photo-tiles only.

## Usage

```bash
pip install -r mosaica/requirements.txt

# color-block mode
python -m mosaica.cli --mode color --target in.jpg --output out.png --tile-width 10 --tile-height 10

# photo mode
python -m mosaica.cli --mode photo --target in.jpg --output out.png \
    --codebook-dir photos/ --tile-width 16 --tile-height 16 --avoid-repeat-within 8
```

### Inspecting embeddings directly

```python
from PIL import Image
from mosaica.models.pretrained_embedding import PretrainedEmbeddingModel

model = PretrainedEmbeddingModel()
vec = model.embed_image(Image.open("some_photo.jpg").convert("RGB"))

print(vec.shape)   # (512,)
```

## Tests

```bash
# from the project root (the folder containing mosaica/)
PYTHONPATH=. python3 mosaica/tests/test_smoke.py
```

No pytest dependency — the script runs standalone and prints pass/fail per check.

> **Windows/PYTHONPATH note:** if you get `ModuleNotFoundError: No module named 'mosaica'`, you're likely running the script from inside a subfolder. Run it from the project root, or set `PYTHONPATH` to the root explicitly (e.g. `PYTHONPATH=/d python mosaica/tests/test_smoke.py` from Git Bash on Windows).

## Requirements

```
pillow
numpy
scipy
torch
torchvision
```

`torch`/`torchvision` are only needed for `mosaica.models` (embedding inspection, and Phase 3 onward) — `color` and `photo` modes work without them.