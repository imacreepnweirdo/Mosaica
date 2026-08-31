"""
Photo-tile rendering — each cell is replaced by a real codebook image,
chosen by a Matcher (color-based today, embedding-based later).
"""

from mosaica.rendering.base import Renderer


class PhotoTileRenderer(Renderer):
    def __init__(self, matcher, tiles):
        """
        matcher: a fitted NearestNeighborMatcher (matcher.fit(tiles) already called)
        tiles: the same list of PIL Images the matcher was fit on
        """
        self.matcher = matcher
        self.tiles = tiles

    def render(self, region_image):
        idx = self.matcher.query(region_image)
        return self.tiles[idx]
