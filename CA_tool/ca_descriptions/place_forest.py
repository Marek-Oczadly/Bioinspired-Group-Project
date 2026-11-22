"""Decides where extra forest goes before the simulation begins.

forest_fires.py calls place_forest() once, on the default stage, and does
everything with the answer (including skipping cells that can't take forest,
such as the lake). This file only chooses the cells, so a smarter strategy can
replace the random one without touching the fire model.
"""

import numpy as np

# For now the forest is a fixed number of random tiles
RANDOM_FOREST_SQUARES = 50


def place_forest(stage, rng=None):
    """Choose the cells to turn into forest.

    stage: the default stage, a list of rows of Cells, before anything has
        been placed or lit. It must not be modified. The random strategy
        ignores what is in it, but a smarter one will want to see the terrain.
    rng: a numpy Generator, so a run can be repeated exactly; omit for a
        different choice every time.

    There is no limit on how many cells, and this does not avoid the lake: the
    caller discards any cell that can't take forest.

    Returns an integer array of shape (number of cells, 2) of (row, col)
    co-ordinates, with no cell repeated.
    """
    rows, cols = len(stage), len(stage[0])
    count = min(RANDOM_FOREST_SQUARES, rows * cols)
    rng = np.random.default_rng() if rng is None else rng

    # Choosing flat positions without replacement guarantees unique cells
    flat_positions = rng.choice(rows * cols, size=count, replace=False)
    return np.column_stack(np.unravel_index(flat_positions, (rows, cols)))
