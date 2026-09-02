"""
Splits a target image into a grid of cells. This is the one piece of logic
every mosaic mode needs, regardless of what ends up drawn in each cell.
"""

from dataclasses import dataclass
from PIL import Image


@dataclass(frozen=True)
class Cell:
    """One grid cell's position, in both grid coordinates and pixel bounds."""

    gx: int
    gy: int
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top


def grid_dimensions(image_width, image_height, cell_width, cell_height):
    """How many whole cells fit across and down the image."""

    grid_width = image_width // cell_width
    grid_height = image_height // cell_height
    return grid_width, grid_height


def iter_cells(image_width, image_height, cell_width, cell_height):
    """
    Yield every Cell in the grid, left-to-right, top-to-bottom.
    Any leftover partial row/column at the edges is simply dropped, matching
    the behavior of the original pixel-mosaic script.
    """

    grid_width, grid_height = grid_dimensions(image_width, image_height, cell_width, cell_height)

    for gy in range(grid_height):
        for gx in range(grid_width):
            left = gx * cell_width
            top = gy * cell_height
            yield Cell(gx, gy, left, top, left + cell_width, top + cell_height)


def crop_cell(image: Image.Image, cell: Cell) -> Image.Image:
    """Crop out the region of `image` that a given Cell covers."""
    return image.crop((cell.left, cell.top, cell.right, cell.bottom))


def canvas_size(image_width, image_height, cell_width, cell_height):
    """Pixel size of the output mosaic (grid_width*cell_width, grid_height*cell_height)."""
    grid_width, grid_height = grid_dimensions(image_width, image_height, cell_width, cell_height)
    return grid_width * cell_width, grid_height * cell_height
