"""
Decides where extra forest goes before the simulation begins. Edit the place_forest() function to change the strat
Atm it places a square radius 10 around hte power plant and incinerator and 3 around the town
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
    # (row, col) of every cell whose special type is named in type_names.
    return [(row, col)
            for row, cells in enumerate(stage)
            for col, cell in enumerate(cells)
            if cell.special_type is not None
            and cell.special_type.name in type_names]


def mark_block(chosen, centre, radius):
    
    # square radius around the centre cell
    row, col = centre
    chosen[max(0, row - radius):row + radius + 1,
           max(0, col - radius):col + radius + 1] = True


def place_forest(stage, rng=None):
    # This simply places them around ignition sources and the town
    
    chosen = np.zeros((len(stage), len(stage[0])), dtype=bool)

    for source in find_cells_of_type(stage, IGNITION_TYPES):
        mark_block(chosen, source, SOURCE_BLOCK_RADIUS)

    # Every town cell plus the margin round it is the same as a block of the
    # margin's size on each town cell
    for town_cell in find_cells_of_type(stage, TOWN_TYPE):
        mark_block(chosen, town_cell, TOWN_MARGIN)

    return np.argwhere(chosen)
