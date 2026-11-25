"""Decides where the water goes.

forest_fires.py calls place_water() once, when the response arrives, and does
everything else (working out how many cells the water covers, wetting the
cells, putting out fires, drying). This file only chooses the cells.

Strategy: water on a burning cell puts it out (it becomes burnt, which can't
spread fire), so if every burning cell is covered the fire is stopped for good.
The water goes on the burning cells nearest the town first, so that if there
are more fires than water, the ones left burning are the furthest from the
town. Any water left over wets the unburnt cells around the fire, again
nearest the town first, as a firebreak. Water the fire doesn't need is not
dropped at all.
"""

import numpy as np

# The 8 cells around a cell, as (row, col) offsets
NEIGHBOUR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                     (0, 1), (1, -1), (1, 0), (1, 1)]


def cells_nearest(candidates, target):
    """The (row, col) rows of `candidates` ordered by distance to `target`."""
    distances = np.hypot(candidates[:, 0] - target[0],
                         candidates[:, 1] - target[1])
    return candidates[np.argsort(distances, kind="stable")]


def unburnt_cells_around(burning, shape):
    """Boolean mask of the cells touching a burning cell, but not burning."""
    around = np.zeros(shape, dtype=bool)
    rows, cols = shape
    for row_offset, col_offset in NEIGHBOUR_OFFSETS:
        # The part of the board whose shifted copy still lies on the board
        source = (slice(max(0, -row_offset), rows - max(0, row_offset)),
                  slice(max(0, -col_offset), cols - max(0, col_offset)))
        shifted = (slice(max(0, row_offset), rows - max(0, -row_offset)),
                   slice(max(0, col_offset), cols - max(0, -col_offset)))
        around[shifted] |= burning[source]
    return around & ~burning


def place_water(board, num_squares, rng=None, burning=None, town=None):
    """Choose the cells to put water on.

    board: the board state, a 2D array with one number per cell (the grid
        from the CA). It must not be modified.
    num_squares: the most cells to choose. How much water there is, and so
        how many cells it covers, is the caller's job; this only decides where
        they are. Fewer cells are returned if less water is needed (it does
        not have to use the full budget).
    rng: a numpy Generator, so a run can be repeated exactly; omit for a
        different choice every time. Used only when `burning` isn't given.
    burning: boolean mask the shape of the board, True where a cell is on
        fire. Without it nothing is known about the fire, so the water is
        placed at random (and does use the full budget).
    town: boolean mask the shape of the board, True where the town is. Used
        to put the water nearest the town first. Without it, the centre of
        the board is used.

    Returns an integer array of shape (number of cells, 2) of (row, col)
    co-ordinates, with no cell repeated. It is empty if there is no fire. Some may be cells that can't take
    water (a lake); the caller ignores those.
    """
    rows, cols = board.shape
    count = min(num_squares, rows * cols)
    rng = np.random.default_rng() if rng is None else rng

    if burning is None:
        # Nothing known about the fire: choose flat positions without
        # replacement, which guarantees unique cells
        flat_positions = rng.choice(rows * cols, size=count, replace=False)
        return np.column_stack(np.unravel_index(flat_positions, board.shape))

    if town is not None and town.any():
        town_centre = np.argwhere(town).mean(axis=0)
    else:
        town_centre = np.array([(rows - 1) / 2, (cols - 1) / 2])

    # Put out the fire nearest the town first, then wet the cells around the
    # fire (nearest the town first) with whatever water is left
    on_fire = cells_nearest(np.argwhere(burning), town_centre)
    around_fire = cells_nearest(
        np.argwhere(unburnt_cells_around(burning, board.shape)), town_centre)
    # The water doesn't have to be used up: if the fire and the cells around
    # it need less than `count`, the rest is simply not dropped
    return np.concatenate([on_fire, around_fire])[:count]
