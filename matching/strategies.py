"""
AvoidanceStrategy implementations — how to pick among several ranked
candidate matches.
"""

from collections import deque
from mosaica.matching.base import AvoidanceStrategy


class NoAvoidance(AvoidanceStrategy):
    """Always take the single best match. Simple, but repeats a lot on flat regions."""

    def choose(self, ranked_candidate_ids):
        return ranked_candidate_ids[0]


class CooldownAvoidance(AvoidanceStrategy):
    """
    Skip any candidate used within the last `window` picks; fall back to the
    best match if every candidate in the ranked list is on cooldown.
    """

    def __init__(self, window=6):
        self.window = window
        self._recent = deque(maxlen=window) if window > 0 else None

    def reset(self):
        if self.window > 0:
            self._recent = deque(maxlen=self.window)

    def choose(self, ranked_candidate_ids):
        if not self._recent:
            chosen = ranked_candidate_ids[0]
        else:
            chosen = next(
                (idx for idx in ranked_candidate_ids if idx not in self._recent),
                ranked_candidate_ids[0],
            )

        if self._recent is not None:
            self._recent.append(chosen)

        return chosen
