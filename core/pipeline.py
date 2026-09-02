"""
The one pipeline every mosaic mode runs through. Modes differ only in which
Renderer gets passed in — this function has no idea whether it's making a
color-block mosaic or a photo mosaic.
"""

from PIL import Image
from mosaica.core import grid


def run_mosaic(target_path, output_path, renderer, tile_width, tile_height, enlargement=1):
    target = Image.open(target_path).convert("RGB")

    if enlargement != 1:
        target = target.resize(
            (target.width * enlargement, target.height * enlargement), Image.LANCZOS
        )

    out_width, out_height = grid.canvas_size(target.width, target.height, tile_width, tile_height)
    output = Image.new("RGB", (out_width, out_height))

    grid_width, grid_height = grid.grid_dimensions(target.width, target.height, tile_width, tile_height)
    print(f"[pipeline] target {target.width}x{target.height} -> grid {grid_width}x{grid_height}")

    for cell in grid.iter_cells(target.width, target.height, tile_width, tile_height):
        region = grid.crop_cell(target, cell)
        patch = renderer.render(region)
        output.paste(patch, (cell.left, cell.top))

        if cell.gx == 0 and cell.gy % 10 == 0:
            print(f"  row {cell.gy}/{grid_height}")

    output.save(output_path)
    print(f"[pipeline] saved -> {output_path}")
    return output
