"""Decides where the water goes.

forest_fires.py calls place_water() once, when the response arrives, and does
everything else (working out how many cells the water covers, wetting the
cells, putting out fires, drying). This file only chooses the cells, so a
smarter strategy can replace the random one without touching the fire model.
"""

import numpy as np


def place_water(board, num_squares, rng=None):
    """Choose the cells to put water on.

    board: the board state, a 2D array with one number per cell (the grid
        from the CA). It must not be modified. The random strategy ignores
        it, but a smarter one will want to see where the fire is.
    num_squares: how many cells to choose. How much water there is, and so
        how many cells it covers, is the caller's job; this only decides where
        they are. (If the board has fewer cells than this, every cell is
        returned, since a cell can't be chosen twice.)
    rng: a numpy Generator, so a run can be repeated exactly; omit for a
        different choice every time.

    Returns an integer array of shape (number of cells, 2) of (row, col)
    co-ordinates, with no cell repeated. Some may be cells that can't take
    water (a lake); the caller ignores those.
    """
    rows, cols = board.shape
    count = min(num_squares, rows * cols)
    rng = np.random.default_rng() if rng is None else rng

    # Choosing flat positions without replacement guarantees unique cells
    flat_positions = rng.choice(rows * cols, size=count, replace=False)
    return np.column_stack(np.unravel_index(flat_positions, board.shape))
