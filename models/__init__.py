"""Model wrappers used in later phases of the mosaic pipeline.

Phase 2 keeps this intentionally small: a pretrained CNN wrapper that exposes a
single, easy-to-inspect embedding vector for one image. The goal is to study the
shape and semantics of the representation before it is plugged into nearest-
neighbor matching in Phase 3.
"""

from mosaica.models.pretrained_embedding import PretrainedEmbeddingModel

__all__ = ["PretrainedEmbeddingModel"]
