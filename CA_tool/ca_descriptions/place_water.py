"""
Decides where the water goes.
Atm it chooses burning cells nearest the town first, then the unburnt cells around them. There is a maximum of 50 but that has never been reached

"""

import numpy as np

# The 8 cells around a cell, as (row, col) offsets
NEIGHBOUR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                     (0, 1), (1, -1), (1, 0), (1, 1)]


def cells_nearest(candidates, target):
    # The (row, col) rows of candidates ordered by distance to target
    distances = np.hypot(candidates[:, 0] - target[0],
                         candidates[:, 1] - target[1])
    return candidates[np.argsort(distances, kind="stable")]


def unburnt_cells_around(burning, shape):
    #Boolean mask of the cells touching a burning cell, but not burning.
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

    board: the board state, 2d np array do not edit
    num_squares: The number of budget of squares to place water on. Not enforced in the function
    rng: a numpy Generator
    burning: boolean mask the shape of the board, True where a cell is on
        fire.
    town: boolean mask the shape of the board, True where the town is so thats the priority
    
    Ran in the forest_fires.py
    """
    rows, cols = board.shape
    count = min(num_squares, rows * cols)
    rng = np.random.default_rng() if rng is None else rng

    if burning is None: # fallback just place randomly
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

    return np.concatenate([on_fire, around_fire])[:count]
