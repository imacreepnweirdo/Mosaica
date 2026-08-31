"""
Generic nearest-neighbor tile matching.

The design deliberately separates two concerns that are easy to conflate:

  1. FeatureExtractor  — turns an image into a vector ("what does it look
                          like, numerically?"). Swappable: mean color today,
                          a CNN embedding later. This is the only thing that
                          changes between color-matching and embedding-matching.

  2. AvoidanceStrategy — given a ranked list of candidate matches, decides
                          which one to actually use (e.g. skip ones used too
                          recently). Independent of what kind of feature
                          vector produced the ranking.

NearestNeighborMatcher combines any FeatureExtractor + any AvoidanceStrategy
into a working matcher. color_matcher.py and (later) embedding_matcher.py are
just convenience constructors around this one class.
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial import cKDTree


class FeatureExtractor(ABC):
    """Turns images into feature vectors used for similarity search."""

    @abstractmethod
    def extract_batch(self, images):
        """images: list of PIL Images -> (N, D) numpy array."""
        raise NotImplementedError

    def extract_one(self, image):
        """Convenience wrapper for a single image -> (D,) numpy array."""
        return self.extract_batch([image])[0]


class AvoidanceStrategy(ABC):
    """Decides which candidate (of several ranked matches) to actually use."""

    @abstractmethod
    def choose(self, ranked_candidate_ids):
        """ranked_candidate_ids: list of tile indices, best match first."""
        raise NotImplementedError

    def reset(self):
        pass


class NearestNeighborMatcher:
    """
    Fits a FeatureExtractor over a codebook of tiles, then finds the closest
    tile to any query region, using an AvoidanceStrategy to pick among the
    top-k closest when the single best match isn't a good idea (e.g. reused
    too recently).
    """

    def __init__(self, feature_extractor: FeatureExtractor, avoidance: AvoidanceStrategy, lookahead_k=8):
        self.feature_extractor = feature_extractor
        self.avoidance = avoidance
        self.lookahead_k = lookahead_k
        self._tree = None
        self._n_tiles = 0

    def fit(self, tiles):
        """tiles: list of PIL Images (the codebook, already tile-sized)."""
        features = self.feature_extractor.extract_batch(tiles)
        self._tree = cKDTree(features)
        self._n_tiles = len(tiles)
        self.avoidance.reset()

    def query(self, region_image):
        """region_image: a PIL Image cropped from the target -> best tile index."""
        if self._tree is None:
            raise RuntimeError("Matcher.fit() must be called before query()")

        k = min(self.lookahead_k, self._n_tiles)
        feature = self.feature_extractor.extract_one(region_image)
        _, candidate_idxs = self._tree.query(feature, k=k)
        candidate_idxs = np.atleast_1d(candidate_idxs).tolist()

        return self.avoidance.choose(candidate_idxs)
