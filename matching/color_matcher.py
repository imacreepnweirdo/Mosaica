"""
Mean-RGB-color matching — the same technique as the original prototype,
now expressed as a FeatureExtractor plugged into the generic matcher.
"""

import numpy as np
from mosaica.matching.base import FeatureExtractor, NearestNeighborMatcher
from mosaica.matching.strategies import CooldownAvoidance, NoAvoidance


class ColorFeatureExtractor(FeatureExtractor):
    """Feature vector = the image's average (R, G, B)."""

    def extract_batch(self, images):
        return np.array([np.array(img.getdata()).mean(axis=0) for img in images])


def build_color_matcher(avoid_repeat_within=6, lookahead_k=8):
    """Convenience constructor: everything needed for color-based tile matching."""

    avoidance = CooldownAvoidance(avoid_repeat_within) if avoid_repeat_within > 0 else NoAvoidance()
    return NearestNeighborMatcher(
        feature_extractor=ColorFeatureExtractor(),
        avoidance=avoidance,
        lookahead_k=max(lookahead_k, avoid_repeat_within + 1),
    )
