"""Sample cells for the random portion of a water drop.

The forest-fire model reserves manual coordinates and excludes lakes before
sampling, then applies the water and updates the board.
"""

import numpy as np


def place_water(board, num_squares, rng=None, *, eligible=None):
    """Sample up to num_squares cells uniformly without replacement.

    board: current grid. Used for its shape.
    num_squares: maximum number of cells to select.
    rng: optional NumPy generator for reproducible placement.
    eligible: optional Boolean mask with the same shape as the board.
        True entries can be selected. If omitted, all cells are eligible.

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
