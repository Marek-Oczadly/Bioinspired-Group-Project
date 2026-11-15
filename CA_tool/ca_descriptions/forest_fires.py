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

import math
import os
import numpy as np
from capyle.ca import Grid2D, Neighbourhood, randomise2d
import capyle.utils as utils
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

# Edit these to change which cells the burning starts from.
BURN_INCINERATOR: bool = True
BURN_POWER_PLANT: bool = True

# Direction the wind comes FROM, in degrees clockwise from north (the usual
# meteorological convention): 0 = northerly, 90 = easterly, 180 = southerly,
# 270 = westerly. The reference map shows a northerly wind.
WIND_DIRECTION: float = 0.0

# Wind speed in km/h. TODO: placeholder value, replace with the real one.
WIND_SPEED: float = 10.0

# The forward spread rate of the fire is about 10% of the wind speed in km/h.
WIND_SPREAD_FRACTION: float = 0.1


def wind_components(direction):
    """Convert a wind direction (degrees, as WIND_DIRECTION) into two
    perpendicular components, each between -1 and 1, for the way the wind
    is BLOWING TOWARDS:

        east:  +1 blowing fully east,  -1 fully west
        north: +1 blowing fully north, -1 fully south

    A northerly wind (0 degrees) blows south, giving (east=0, north=-1).
    Any float is accepted; it is wrapped into 0-360. Note grid row 0 is the
    north edge of the stage, so "north" points towards decreasing row index.
    """
    radians = math.radians(direction % 360)
    east = -math.sin(radians)
    north = -math.cos(radians)
    return east, north


# The 8 neighbours in the order Grid2D hands them to the transition function
# (neighbourstates: NW, N, NE, W, E, SW, S, SE), as the (east, north) offset
# of each neighbour from the cell.
MOORE_OFFSETS = ((-1, 1), (0, 1), (1, 1),
                     (-1, 0), (1, 0),
                     (-1, -1), (0, -1), (1, -1))


def wind_multipliers(east, north, speed, offsets=MOORE_OFFSETS):
    """How much the wind boosts fire spread into a cell from each neighbour.

    east, north: the wind components from wind_components().
    speed: the wind speed in km/h (as WIND_SPEED).
    offsets: the neighbour arrangement, (east, north) offsets of each
        neighbour from the cell (default: MOORE_OFFSETS, which matches the
        order of neighbourstates, so index k lines up with
        neighbourstates[k]).
    """
    forward_spread_rate = WIND_SPREAD_FRACTION * speed
    multipliers = []
    for offset_east, offset_north in offsets:
        length = math.hypot(offset_east, offset_north)
        # Fire travels from the neighbour to the cell: opposite to the offset
        travel_east = -offset_east / length
        travel_north = -offset_north / length
        alignment = travel_east * east + travel_north * north
        multipliers.append(forward_spread_rate * max(0.0, round(alignment, 10)))  # ReLU, then scale
    multipliers = np.array(multipliers)
    multipliers.setflags(write=False)
    return multipliers


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


# The ignition points for the fire. Edit the bools at the top rather than this
IGNITION_POINTS = tuple(
    point for point, burn in ((SpecialType.INCINERATOR, BURN_INCINERATOR),
                              (SpecialType.POWER_PLANT, BURN_POWER_PLANT))
    if burn
)


# Colours are sampled from the reference map (forest_fires_meta/town.png).
TERRAIN_COLOURS = {
    Terrain.CHAPARRAL: (0.75, 0.76, 0.24),
    Terrain.DENSE_FOREST: (0.31, 0.38, 0.18),
    Terrain.SCRUBLAND_CANYON: (1.0, 1.0, 0.5),
    Terrain.LAKE: (0.0, 0.69, 0.94),
}

# Special cells are drawn on top of their terrain, in grey.
SPECIAL_COLOURS = {
    SpecialType.TOWN: (0.5, 0.5, 0.5),
    SpecialType.INCINERATOR: (0.5, 0.5, 0.5),
    SpecialType.POWER_PLANT: (0.5, 0.5, 0.5),
}

# A burning cell is a strong orange, shifted per terrain so you can still tell
# what is burning: darker/redder for dense forest, yellower for scrubland.
BURNING_COLOURS = {
    Terrain.CHAPARRAL: (1.0, 0.55, 0.05),
    Terrain.DENSE_FOREST: (0.85, 0.28, 0.0),
    Terrain.SCRUBLAND_CANYON: (1.0, 0.72, 0.0),
    Terrain.LAKE: (1.0, 0.45, 0.35),  # lakes should not burn; here for completeness
}

# A burnt cell is a blackened version of its terrain colour: the terrain
# colour scaled down so it is still recognisably that tile.
BURNT_DARKNESS = 0.35


def blacken(colour):
    """Darken an RGB colour towards black."""
    return tuple(channel * BURNT_DARKNESS for channel in colour)


BURNT_COLOURS = {terrain: blacken(colour)
                 for terrain, colour in TERRAIN_COLOURS.items()}

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


def begin_ignition(stage, ignition_points):
    """Set every cell whose special type is in ignition_points burning."""
    for row in stage:
        for cell in row:
            if cell.special_type in ignition_points:
                cell.burn_state = BurnState.BURNING


STAGE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "forest_fires_meta", "town.txt")

# Each distinct look gets one key:
#   Terrain / SpecialType   an unburnt cell
#   (Terrain, BURNING)      a burning cell, coloured per terrain
#   (Terrain, BURNT)       a burnt cell, a blackened version of its terrain
DISPLAY_COLOURS = {}
DISPLAY_COLOURS.update(TERRAIN_COLOURS)
DISPLAY_COLOURS.update(SPECIAL_COLOURS)
DISPLAY_COLOURS.update({(terrain, BurnState.BURNING): colour
                        for terrain, colour in BURNING_COLOURS.items()})
DISPLAY_COLOURS.update({(terrain, BurnState.BURNT): colour
                        for terrain, colour in BURNT_COLOURS.items()})

DISPLAY_KEYS = list(DISPLAY_COLOURS)
STATES = list(range(len(DISPLAY_KEYS)))  # number for each
STATE_COLOURS = [DISPLAY_COLOURS[key] for key in DISPLAY_KEYS]  # colour for each


def display_state(cell):
    """The numeric grid state used to draw a cell. Burning and burnt cells
    show their burn colour; an unburnt special type (town, incinerator, power
    plant) hides the terrain underneath it."""
    if cell.burn_state != BurnState.UNBURNT:
        key = (cell.terrain, cell.burn_state)
    elif cell.special_type is not None:
        key = cell.special_type
    else:
        key = cell.terrain
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
    stage = load_stage(STAGE_PATH)
    begin_ignition(stage, IGNITION_POINTS)
    initial_grid = stage_to_grid(stage)
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


def transition_function(grid, neighbourstates, neighbourcounts, multipliers):
    """Function to apply the transition rules
    and return the new grid. multipliers are the wind multipliers from
    wind_multipliers(), one per neighbour in the order of neighbourstates."""
    # TODO: That actual transition logic currently its placeholder
    return grid


def main():
    """ Main function that sets up, runs and saves CA"""
    # Get the config object from set up
    config = setup(sys.argv[1:])

    # The wind is fixed for the whole run, so work out the multipliers once
    # here and hand them to every call of the transition function.
    multipliers = wind_multipliers(*wind_components(WIND_DIRECTION),
                                   WIND_SPEED)

    # Create grid object using parameters from config + transition function
    grid = Grid2D(config, (transition_function, multipliers))

    # Run the CA, save grid state every generation to timeline
    timeline = grid.run()

    # Save updated config to file
    config.save()
    # Save timeline to file
    utils.save(timeline, config.timeline_path)

if __name__ == "__main__":
    main()
