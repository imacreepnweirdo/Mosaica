"""
CNN-embedding-based matching. Same interface as color matching
(ColorFeatureExtractor), but each image is represented by a 512-dimensional
ResNet feature vector instead of 3 mean-RGB numbers — so matches reflect
texture, shape, and structure, not just average color.

The cosine similarity trick
----------------------------
Cosine similarity compares the *direction* two vectors point in, not their
distance — that's what makes it the right comparison for CNN embeddings (see
the Phase 2 notes in the root README). But the search structure we're
already using, scipy's cKDTree, only supports plain (euclidean) distance.

Rather than build a second search structure, we use a standard trick:
L2-normalize every vector to unit length before indexing. For unit vectors,
euclidean distance and cosine similarity are related by a fixed formula:

    ||u - v||^2 = 2 - 2 * cosine_similarity(u, v)

Squared euclidean distance goes *down* exactly when cosine similarity goes
*up*. So the nearest neighbor by euclidean distance among unit vectors is
guaranteed to be the nearest neighbor by cosine similarity too — we get
cosine-similarity search for free out of the same cKDTree used for color
matching, just by normalizing first.
"""

import numpy as np

from mosaica.matching.base import FeatureExtractor, NearestNeighborMatcher
from mosaica.matching.strategies import CooldownAvoidance, NoAvoidance
from mosaica.models.pretrained_embedding import PretrainedEmbeddingModel


def _l2_normalize(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1  # guard against an all-zero embedding causing a divide-by-zero
    return features / norms


class EmbeddingFeatureExtractor(FeatureExtractor):
    """Feature vector = L2-normalized pretrained-CNN embedding."""

    def __init__(self, model: PretrainedEmbeddingModel = None):
        # Accepting a pre-built model lets tests/scripts reuse one instance
        # instead of reloading the network weights repeatedly.
        self.model = model or PretrainedEmbeddingModel()

    def extract_batch(self, images):
        raw = self.model.embed_batch(images)
        return _l2_normalize(raw)


def build_embedding_matcher(avoid_repeat_within=6, lookahead_k=8, model=None):
    """Convenience constructor: everything needed for embedding-based tile matching."""

    avoidance = CooldownAvoidance(avoid_repeat_within) if avoid_repeat_within > 0 else NoAvoidance()
    return NearestNeighborMatcher(
        feature_extractor=EmbeddingFeatureExtractor(model=model),
        avoidance=avoidance,
        lookahead_k=max(lookahead_k, avoid_repeat_within + 1),
    )
