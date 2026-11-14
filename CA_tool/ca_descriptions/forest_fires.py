# Name: Forest Fires
# Dimensions: 2

# --- Set up executable path, do not edit ---
import sys
import inspect
this_file_loc = (inspect.stack()[0][1])
main_dir_loc = this_file_loc[:this_file_loc.index('ca_descriptions')]
sys.path.append(main_dir_loc)
sys.path.append(main_dir_loc + 'capyle')
sys.path.append(main_dir_loc + 'capyle/ca')
sys.path.append(main_dir_loc + 'capyle/guicomponents')
# ---

import os
import numpy as np
from capyle.ca import Grid2D, Neighbourhood, randomise2d
import capyle.utils as utils
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple


class Terrain(Enum):
    CHAPARRAL = auto()
    DENSE_FOREST = auto()
    SCRUBLAND_CANYON = auto()
    LAKE = auto()


class BurnState(Enum):
    UNBURNT = auto()
    BURNING = auto()
    BURNT = auto()


class SpecialType(Enum):
    TOWN = auto()
    INCINERATOR = auto()
    POWER_PLANT = auto()


# Colours are sampled from the reference map (forest_fires_meta/town.png).
TERRAIN_COLOURS = {
    Terrain.CHAPARRAL: (0.75, 0.76, 0.24),
    Terrain.DENSE_FOREST: (0.31, 0.38, 0.18),
    Terrain.SCRUBLAND_CANYON: (1.0, 1.0, 0.5),
    Terrain.LAKE: (0.0, 0.69, 0.94),
}

# Special cells are drawn on top of their terrain. The reference map marks
# the town, power plant and incinerator in black.
SPECIAL_COLOURS = {
    SpecialType.TOWN: (0.0, 0.0, 0.0),
    SpecialType.INCINERATOR: (0.0, 0.0, 0.0),
    SpecialType.POWER_PLANT: (0.0, 0.0, 0.0),
}

# TODO: placeholder values for the burn properties of every terrain type,
# replace with real ones
TERRAIN_DEFAULTS = {
    terrain: dict(
        colour=TERRAIN_COLOURS[terrain],
        burn_probability=0.0,
        max_burn_duration=0,
        water_fraction=0.0,
    )
    for terrain in Terrain
}


@dataclass
class Cell:
    terrain: Terrain
    colour: Tuple[float, float, float]  # RGB, each 0-1 (CAPyLE state_colors format)
    burn_probability: float
    remaining_burn_duration: float
    max_burn_duration: float
    special_type: Optional[SpecialType] = None  # None, TOWN, INCINERATOR, or POWER_PLANT
    water_fraction: float = 0.0
    burn_state: BurnState = BurnState.UNBURNT

    @classmethod
    def from_terrain(cls, terrain, **overrides):
        """Create a cell using the (placeholder) defaults for its terrain.
        Any field can be overridden by keyword."""
        values = dict(TERRAIN_DEFAULTS[terrain])
        values.setdefault("remaining_burn_duration", values["max_burn_duration"])
        values.update(overrides)
        return cls(terrain=terrain, **values)


TERRAIN_CODES = {
    "C": Terrain.CHAPARRAL,
    "D": Terrain.DENSE_FOREST,
    "S": Terrain.SCRUBLAND_CANYON,
    "L": Terrain.LAKE,
}

SPECIAL_CODES = {
    "T": SpecialType.TOWN,
    "I": SpecialType.INCINERATOR,
    "P": SpecialType.POWER_PLANT,
    "N": None,
}


def load_stage(path):
    """Create a stage (a list of rows, each a list of Cells) from a text file.

    Format: row major, cells separated by spaces, rows separated by newlines.
    Each cell is two characters: a terrain code (C, D, S, L) followed by a
    special type code (T, I, P, N for none), e.g. "DN LN CT".
    All cells start unburnt. Blank lines are ignored.

    Raises ValueError on an unknown code or if rows differ in length.
    """
    stage = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            tokens = line.split()
            if not tokens:
                continue
            row = []
            for token in tokens:
                if (len(token) != 2 or token[0] not in TERRAIN_CODES
                        or token[1] not in SPECIAL_CODES):
                    raise ValueError(
                        "Invalid cell {!r} on line {}".format(token, line_num))
                row.append(Cell.from_terrain(
                    TERRAIN_CODES[token[0]],
                    special_type=SPECIAL_CODES[token[1]]))
            if stage and len(row) != len(stage[0]):
                raise ValueError(
                    "Line {} has {} cells, expected {}".format(
                        line_num, len(row), len(stage[0])))
            stage.append(row)
    return stage


STAGE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "forest_fires_meta", "town.txt")

DISPLAY_KEYS = list(Terrain) + list(SpecialType)
STATES = list(range(len(DISPLAY_KEYS)))  # number for each
STATE_COLOURS = ([TERRAIN_COLOURS[terrain] for terrain in Terrain]
                 + [SPECIAL_COLOURS[special] for special in SpecialType]) # colour for each


def display_state(cell):
    """The numeric grid state used to draw a cell. A special type (town,
    incinerator, power plant) hides the terrain underneath it."""
    key = cell.special_type if cell.special_type is not None else cell.terrain
    return DISPLAY_KEYS.index(key)


def stage_to_grid(stage):
    """Convert a stage (list of rows of Cells) to a numeric numpy grid."""
    return np.array([[display_state(cell) for cell in row] for row in stage])


def setup(args):
    """Set up the config object used to interact with the GUI"""
    config_path = args[0]
    config = utils.load(config_path)
    # -- THE CA MUST BE RELOADED IN THE GUI IF ANY OF THE BELOW ARE CHANGED --
    config.title = "Forest Fires"
    config.dimensions = 2
    config.states = STATES
    # -------------------------------------------------------------------------

    # ---- Override the defaults below (these may be changed at anytime) ----

    config.state_colors = STATE_COLOURS
    initial_grid = stage_to_grid(load_stage(STAGE_PATH))
    config.grid_dims = initial_grid.shape
    config.initial_grid = initial_grid

    # ----------------------------------------------------------------------

    # the GUI calls this to pass the user defined config
    # into the main system with an extra argument
    # do not change
    if len(args) == 2:
        config.save()
        sys.exit()
    return config


def transition_function(grid, neighbourstates, neighbourcounts):
    """Function to apply the transition rules
    and return the new grid"""
    # TODO: That actual transition logic currently its placeholder
    return grid


def main():
    """ Main function that sets up, runs and saves CA"""
    # Get the config object from set up
    config = setup(sys.argv[1:])

    # Create grid object using parameters from config + transition function
    grid = Grid2D(config, transition_function)

    # Run the CA, save grid state every generation to timeline
    timeline = grid.run()

    # Save updated config to file
    config.save()
    # Save timeline to file
    utils.save(timeline, config.timeline_path)

if __name__ == "__main__":
    main()
