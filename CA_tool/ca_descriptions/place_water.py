"""Decides where the water goes.

forest_fires.py calls place_water() once, when the response arrives, and does
everything else (working out how many cells the water covers, wetting the
cells, putting out fires, drying). This file only chooses the cells.

Strategy: water on a burning cell puts it out (it becomes burnt, which can't
spread fire), so if every burning cell is covered the fire is stopped for good.
The water goes on the burning cells nearest the town first, so that if there
are more fires than water, the ones left burning are the furthest from the
town. Any water left over wets the unburnt cells around the fire, again
nearest the town first, as a firebreak.
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
    num_squares: how many cells to choose. How much water there is, and so
        how many cells it covers, is the caller's job; this only decides where
        they are. (If the board has fewer cells than this, every cell is
        returned, since a cell can't be chosen twice.)
    rng: a numpy Generator, so a run can be repeated exactly; omit for a
        different choice every time. Used only to fill in random cells if
        there is nothing to aim at (no fire).
    burning: boolean mask the shape of the board, True where a cell is on
        fire. Without it the water is placed at random.
    town: boolean mask the shape of the board, True where the town is. Used
        to put the water nearest the town first. Without it, the centre of
        the board is used.

    Returns an integer array of shape (number of cells, 2) of (row, col)
    co-ordinates, with no cell repeated. Some may be cells that can't take
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
    chosen = np.concatenate([on_fire, around_fire])[:count]

    if len(chosen) < count:
        # Water is left and there is no fire to use it on: fill up with
        # random cells so the answer is always `count` unique cells
        taken = np.zeros(board.shape, dtype=bool)
        taken[chosen[:, 0], chosen[:, 1]] = True
        free = np.flatnonzero(~taken)
        extra = rng.choice(free, size=count - len(chosen), replace=False)
        chosen = np.concatenate(
            [chosen, np.column_stack(np.unravel_index(extra, board.shape))])
    return chosen
