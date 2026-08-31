"""
A Renderer decides what actually gets drawn into a grid cell. This is the
seam between "color-block mosaic" and "photo mosaic" — the pipeline doesn't
know or care which one it's using.
"""

from abc import ABC, abstractmethod


class Renderer(ABC):
    @abstractmethod
    def render(self, region_image):
        """
        region_image: PIL Image cropped from the target at this cell's location.
        Returns: a PIL Image, tile-sized, to paste in the output at the same position.
        """
        raise NotImplementedError
