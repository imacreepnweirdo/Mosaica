"""
Worked example: inspect what a pretrained CNN embedding actually looks like,
and how to compare two of them.

This is the script used to build intuition for Phase 2 of the project — see
the "Phase 2" section of the root README for what these numbers mean and why
cosine similarity (not raw stats like mean/max) is the right way to compare
two embeddings.

Usage:
    # inspect a single image's embedding
    python -m mosaica.examples.inspect_embedding path/to/photo.jpg

    # compare two images (also runs a self-similarity sanity check)
    python -m mosaica.examples.inspect_embedding path/to/photo1.jpg path/to/photo2.jpg

Run from the project root (the folder containing mosaica/), e.g. on Windows
Git Bash:
    cd /d
    python -m mosaica.examples.inspect_embedding photo1.jpg photo2.jpg
"""

import argparse

import numpy as np
from PIL import Image

from mosaica.models.pretrained_embedding import PretrainedEmbeddingModel


def inspect(name, vec):
    print(f"\n--- {name} ---")
    print("shape:", vec.shape)
    print("dtype:", vec.dtype)
    print("min / max / mean:", vec.min(), vec.max(), vec.mean())
    print("first 10 values:", vec[:10])
    print("how many values are exactly 0:", (vec == 0).sum(), "/", vec.shape[0])


def cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def load_embedding(model, path):
    image = Image.open(path).convert("RGB")
    return model.embed_image(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="Path to an image")
    parser.add_argument("image2", nargs="?", default=None, help="Optional second image, to compare against the first")
    args = parser.parse_args()

    model = PretrainedEmbeddingModel()

    vec1 = load_embedding(model, args.image)
    inspect(args.image, vec1)

    if args.image2 is None:
        return

    vec2 = load_embedding(model, args.image2)
    inspect(args.image2, vec2)

    print("\n--- comparison ---")
    print(f"cosine similarity ({args.image} vs {args.image2}):", cosine_similarity(vec1, vec2))

    # sanity check: an image compared against itself should be ~1.0
    vec1_again = load_embedding(model, args.image)
    print(f"cosine similarity ({args.image} vs itself, sanity check):", cosine_similarity(vec1, vec1_again))


if __name__ == "__main__":
    main()
