# mosaica

A mosaic-generation engine, built as a learning project while picking up PyTorch and CNNs.

Two rendering modes share one pipeline:

- **`color` mode** — each grid cell is redrawn as a flat square of its own average color (a painterly / pixel-art effect). No codebook needed.
- **`photo` mode** — each grid cell is replaced by the closest-matching real photo from a folder you provide (a true photomosaic), using one of three matching strategies (see below).

Inspired by [worldveil/photomosaic](https://github.com/worldveil/photomosaic), but built independently — see [Design notes](#design-notes-vs-the-reference-project).

## Why this project exists

It started as a simple "average the color of a region, draw a square" script, and grew in two directions at once: a real photomosaic (matching regions to real photos, not just flat color), and a vehicle for learning PyTorch/CNNs hands-on by using a pretrained CNN to make that matching smarter than plain color-averaging — and, later, learning what happens when a smarter matcher isn't calibrated correctly.

## Architecture

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
                            (batches cells through renderer.render_batch() when available)
```

- **`FeatureExtractor`** (`matching/base.py`) — turns an image into a feature vector for similarity search:
  - `ColorFeatureExtractor` (`matching/color_matcher.py`) — mean RGB, 3 numbers. Fast, no caching needed.
  - `EmbeddingFeatureExtractor` (`matching/embedding_matcher.py`) — a pretrained ResNet-18's 512-dim feature vector (classifier head removed), L2-normalized so `cKDTree`'s euclidean search doubles as cosine-similarity search (`||u-v||² = 2 - 2·cos_sim(u,v)` for unit vectors). Captures texture/shape, not just color. Expensive — cached per-codebook.
- **`HybridMatcher`** (`matching/hybrid_matcher.py`) — routes each cell to color or embedding matching based on the cell's own pixel variance (flat → color, textured → embedding). See [the calibration story](#a-real-bug-hybrid-matching-calibrated-against-the-wrong-data) below for why this exists and what it took to get right.
- **`AvoidanceStrategy`** (`matching/strategies.py`) — `CooldownAvoidance` skips any tile used within the last *N* picks, so flat regions don't repeat one tile endlessly. In `HybridMatcher`, both underlying matchers **share one avoidance instance**, so a tile placed via one path can't be immediately reused via the other.
- **`Renderer`** (`rendering/base.py`) — `ColorBlockRenderer` and `PhotoTileRenderer`. The pipeline calls `renderer.render(region)` (or `render_batch()`, see below) and has no idea which mode or matcher is running.
- **`NearestNeighborMatcher`** (`matching/base.py`) combines any `FeatureExtractor` + any `AvoidanceStrategy`. Also exposes `query_batch()` — see below.

### Batched inference

`NearestNeighborMatcher.query_batch()`, `HybridMatcher.query_batch()`, and `PhotoTileRenderer.render_batch()` let the pipeline process many cells' feature extraction in one call instead of one cell at a time. `core/pipeline.py` uses this automatically whenever a renderer supports it (`ColorBlockRenderer` doesn't need it and is unaffected). This matters a lot for a CNN: batching is dramatically more efficient than one forward pass per image.

**Guarantee:** batching changes performance, never output. Verified two ways — a unit test confirming `query_batch()` returns the exact same sequence of tile choices as calling `query()` once per image (same codebook, same queries, same avoidance state, checked with both one big batch and several small chunks), and an end-to-end test running the real pipeline twice with different `--batch-size` values and diffing the resulting images pixel-for-pixel.

### Project layout

```
mosaica/
  core/       grid.py, pipeline.py (batched + per-cell rendering), config.py (mode/matcher → Renderer factory)
  codebook/   scanner.py, cache.py, loader.py
  matching/   base.py (interfaces + NearestNeighborMatcher), color_matcher.py, embedding_matcher.py,
              hybrid_matcher.py, strategies.py
  rendering/  base.py, color_block.py, photo_tile.py
  models/     pretrained_embedding.py — ResNet-18 feature extractor
  examples/   inspect_embedding.py, fetch_codebook_images.py
  cli.py      argparse entry point
  tests/      test_smoke.py
```

## Status

- **Phase 1 — Core architecture ✅** Grid splitting, codebook loading with content-hash caching, color and photo modes, cooldown-based repeat avoidance.
- **Phase 2 — PyTorch introduction ✅** `models/pretrained_embedding.py`, a pretrained ResNet-18 with its classifier head replaced by `nn.Identity()`. Learned: embedding values are post-ReLU (≥0), cosine similarity (not raw stats) is the right comparison, verified via a self-similarity sanity check (~1.0) and a real bird-vs-tree comparison (~0.53 — different subjects, shared visual structure).
- **Phase 3 — Embedding-based matching ✅** `EmbeddingFeatureExtractor`, plugged into the existing `NearestNeighborMatcher` with zero changes to the pipeline or CLI, exactly as planned.
- **Phase 3.5 — Hybrid matching, a real calibration bug, and a ~30x speedup ✅** See below — this is the part worth reading in detail.
- **Phase 4 — Analysis tools (planned)** `analysis/` — embedding-space visualization (t-SNE/PCA), mosaic reconstruction quality metric.
- **Phase 5 — Stretch goal** Fine-tune/metric-learn an embedding model on your own codebook.

## A real bug: hybrid matching, calibrated against the wrong data

**The problem.** Pure embedding matching, tested on a real photo (autumn foliage over a lake, with a boat), produced a checkerboard-like artifact across smooth regions — water, sky. Cause: each codebook tile is a whole photo shrunk down, carrying its own internal composition (its own sky corner, its own light/dark split). Color matching happened to find genuinely flat-colored source photos for flat target regions, which blend seamlessly regardless of internal layout. Embedding matching, sensitive to texture, picked photos whose *edges* resembled the target's edges — and tiling many such photos together stacked their shared internal compositions into a visible, repeating pattern that had nothing to do with the target.

**First fix attempt.** Built `HybridMatcher`: route each cell to color or embedding matching based on that cell's own pixel variance — flat cells (low variance) to color, textured cells (high variance) to embedding. Shipped with `variance_threshold=300.0`, calibrated against synthetic solid-color test tiles (variance = exactly 0 for a flat square).

**It didn't work.** Running it on the real photo produced visually the same checkerboard artifact as pure embedding matching — no improvement.

**Diagnosis.** Rather than guess a new number, measured the real per-cell variance distribution directly across the actual target image (63,140 cells at 16×16):

| percentile | variance |
|---|---|
| p25 | 266.6 |
| p50 (median) | 1,117.2 |
| p75 | 2,260.4 |
| p95 | 4,707.8 |

At `threshold=300`, only ~26% of cells in the whole image — and only ~51% of the water region specifically — routed to color matching. Real photos have baseline pixel variance from JPEG compression, ripple shading, and sensor noise that a perfectly flat synthetic test square doesn't; a threshold calibrated on the latter badly underestimated the former. With roughly half the water region's cells still routing to embedding, and alternating cell-by-cell between "smooth color pick" and "high-contrast embedding pick," the checkerboard artifact was structurally guaranteed regardless of whether any individual pick was reasonable in isolation.

**Fix.** Raised the default to `variance_threshold=3000.0` — chosen so ~85% of the whole image's cells route to color matching, concentrating embedding matching on genuinely high-variance regions.

**Result, measured before and after on the same crops:**

| Region | threshold=300 | threshold=3000 |
|---|---|---|
| Water | Checkerboard artifact, unreadable | Smooth — matches pure color-matcher quality |
| Boat | Nearly swallowed by noise | Silhouette clearly legible |
| Branch/foliage | Texture-aware (correct) | Unchanged — still texture-aware (correct) |

**Compounding fix: batching.** Embedding matching was also calling the CNN once per grid cell — one forward pass per image, serially. Added the batched inference path described above.

**Combined measured result**, real 4928×3280 target, 201-photo codebook:

| | Before | After |
|---|---|---|
| Generation time | ~2–3 hours (estimated: unbatched, threshold=300) | **~5 minutes** |
| Cells routed to embedding (CNN) | up to 100% | 9,676 / 63,140 (15.3%) |
| Cells routed to color (cheap) | as low as 0% | 53,464 / 63,140 (84.7%) |

The routing split predicted from the variance measurement *before* the fixed version was ever run (84.7% color / 15.3% embedding) matched the actual measured split (84.68% / 15.32%) almost exactly — the ~30x speedup comes from two compounding effects: ~85% fewer cells need the CNN at all, and the ones that do run in efficient batches instead of one at a time.

**Takeaway:** a smarter matching signal (texture-aware embeddings) can make specific outputs *worse*, not better, if it's applied somewhere it shouldn't be — and the fix for a "matcher A vs matcher B" trade-off often isn't picking a winner, it's routing between them correctly. Also: calibrate thresholds against real data, not synthetic test cases that don't share the real data's noise characteristics.

## Design notes vs. the reference project

[worldveil/photomosaic](https://github.com/worldveil/photomosaic) matches on flattened, resized pixel vectors via `faiss` — still fundamentally color/position data, just higher-resolution than a single average — and spreads near-duplicate logic across several top-level scripts (`mosaic.py`, `video.py`, `interactive.py`, `make_gif.py`).

This project differs on purpose:
- One pipeline with pluggable Renderer/Matcher pieces, not parallel scripts per feature.
- Real CNN feature embeddings (Phase 3), capturing texture/shape/structure — not achievable with flattened pixel vectors alone.
- A hybrid matcher that routes between color and embedding matching per-cell, rather than committing to one matching strategy for the whole image — something neither the reference project nor a naive embedding-only approach does.
- Content-hash-based caching from the start, for both tiles and features.
- Batched inference verified to be output-preserving, not just faster.

## Usage

```bash
pip install -r mosaica/requirements.txt

# color-block mode
python -m mosaica.cli --mode color --target in.jpg --output out.png --tile-width 10 --tile-height 10

# photo mode — color matching (fast)
python -m mosaica.cli --mode photo --target in.jpg --output out.png \
    --codebook-dir photos/ --tile-width 16 --tile-height 16 --matcher color

# photo mode — embedding matching (texture-aware, slower)
python -m mosaica.cli --mode photo --target in.jpg --output out.png \
    --codebook-dir photos/ --tile-width 16 --tile-height 16 --matcher embedding --batch-size 128

# photo mode — hybrid (recommended): color for flat regions, embedding for textured ones
python -m mosaica.cli --mode photo --target in.jpg --output out.png \
    --codebook-dir photos/ --tile-width 16 --tile-height 16 \
    --matcher hybrid --variance-threshold 3000 --batch-size 128
```

Hybrid mode prints routing stats after each run (`color_matched` / `embedding_matched` counts and the variance distribution seen) — use these to retune `--variance-threshold` for a different image if needed; there's nothing universal about `3000`, it's calibrated to this project's specific test image.

### Inspecting embeddings directly

```bash
python -m mosaica.examples.inspect_embedding path/to/photo.jpg
python -m mosaica.examples.inspect_embedding path/to/photo1.jpg path/to/photo2.jpg   # compare two
```

### Building a codebook from free stock photos

```bash
python -m mosaica.examples.fetch_codebook_images --query "nature" --count 200 --out-dir photos
```

Requires a free [Pexels API key](https://www.pexels.com/api/), set as `PEXELS_API_KEY`.

## Tests

```bash
# from the project root (the folder containing mosaica/)
PYTHONPATH=. python3 mosaica/tests/test_smoke.py
```

No pytest dependency. Tests requiring pretrained-model weights (network access) skip gracefully rather than failing if weights can't be downloaded, so an offline/sandboxed environment doesn't hide a real regression in the rest of the suite.

> **Windows/PYTHONPATH note:** if you get `ModuleNotFoundError: No module named 'mosaica'`, run from the project root, or set `PYTHONPATH` explicitly (e.g. `PYTHONPATH=/d python mosaica/tests/test_smoke.py` from Git Bash).

## Requirements

```
pillow
numpy
scipy
torch
torchvision
```

`torch`/`torchvision` are only needed for `mosaica.models` and embedding/hybrid matching — `color` mode and `photo` mode with `--matcher color` work without them.
