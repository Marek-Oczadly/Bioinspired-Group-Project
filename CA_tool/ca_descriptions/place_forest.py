"""Decides where extra forest goes before the simulation begins.

forest_fires.py calls place_forest() once, on the default stage, and does
everything with the answer (including skipping cells that can't take forest,
such as the lake). This file only chooses the cells.

Dense forest spreads fire very slowly, so the forest goes in two places:

1. A square block on every ignition source. It keeps the fire small and slow
   while the water is on its way, so the water can put all of it out. The
   stage is passed in before anything is lit, so we can't know which source
   will burn, and every source gets a block.
2. The town itself and a margin around it, as a second line of defence if some
   fire does get through.
"""

import numpy as np

# How far the forest block reaches from each ignition source, in cells
# (a radius of 10 is a block 11 cells square)
SOURCE_BLOCK_RADIUS = 10

# How many cells of forest to put round the town (the town cells themselves
# are always forest)
TOWN_MARGIN = 3

# Special types that can start a fire (the town is what we protect)
IGNITION_TYPES = ("INCINERATOR", "POWER_PLANT")
TOWN_TYPE = "TOWN"


def find_cells_of_type(stage, type_names):
    """(row, col) of every cell whose special type is named in type_names."""
    return [(row, col)
            for row, cells in enumerate(stage)
            for col, cell in enumerate(cells)
            if cell.special_type is not None
            and cell.special_type.name in type_names]


def mark_block(chosen, centre, radius):
    """Mark the square of cells within `radius` of centre (clipped to the
    board)."""
    row, col = centre
    chosen[max(0, row - radius):row + radius + 1,
           max(0, col - radius):col + radius + 1] = True


def place_forest(stage, rng=None):
    """Choose the cells to turn into forest.

    stage: the default stage, a list of rows of Cells, before anything has
        been placed or lit. It must not be modified.
    rng: a numpy Generator, so a run can be repeated exactly; omit for a
        different choice every time. Not needed: the placement is fixed by
        the stage.

    There is no limit on how many cells, and this does not avoid the lake: the
    caller discards any cell that can't take forest.

    Returns an integer array of shape (number of cells, 2) of (row, col)
    co-ordinates, with no cell repeated.
    """
    chosen = np.zeros((len(stage), len(stage[0])), dtype=bool)

    for source in find_cells_of_type(stage, IGNITION_TYPES):
        mark_block(chosen, source, SOURCE_BLOCK_RADIUS)

    # Every town cell plus the margin round it is the same as a block of the
    # margin's size on each town cell
    for town_cell in find_cells_of_type(stage, TOWN_TYPE):
        mark_block(chosen, town_cell, TOWN_MARGIN)

    return np.argwhere(chosen)
