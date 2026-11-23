"""Decides where the water goes.

forest_fires.py calls place_water() once, when the response arrives, and does
everything else (working out how many cells the water covers, wetting the
cells, putting out fires, drying). This file only chooses the cells, so a
smarter strategy can replace the random one without touching the fire model.
"""

import numpy as np


def place_water(board, num_squares, rng=None, *, eligible=None):
    """Choose the cells to put water on.

    board: current grid. Used for its shape.
    num_squares: maximum number of cells to select.
    rng: optional NumPy generator for reproducible placement.
    eligible: optional Boolean mask of the cells available to select.

    Returns a (number of cells, 2) array of unique [row, column] pairs.
    """
    rows, cols = board.shape
    # Flatten the eligible cells for sampling, then convert back to row/column.
    candidates = (np.arange(rows * cols) if eligible is None
                  else np.flatnonzero(eligible))
    count = min(num_squares, len(candidates))
    rng = np.random.default_rng() if rng is None else rng

    # Choosing flat positions without replacement guarantees unique cells
    flat_positions = rng.choice(candidates, size=count, replace=False)
    return np.column_stack(np.unravel_index(flat_positions, board.shape))
