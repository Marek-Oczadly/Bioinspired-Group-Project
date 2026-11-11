# Name: NAME
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
    GENERATOR = auto()
    POWER_PLANT = auto()


# TODO: placeholder values for every terrain type, replace with real ones
TERRAIN_DEFAULTS = {
    terrain: dict(
        colour=(0.0, 0.0, 0.0),
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
    special_type: Optional[SpecialType] = None  # None, TOWN, GENERATOR, or POWER_PLANT
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


def setup(args):
    """Set up the config object used to interact with the GUI"""
    config_path = args[0]
    config = utils.load(config_path)
    # -- THE CA MUST BE RELOADED IN THE GUI IF ANY OF THE BELOW ARE CHANGED --
    config.title = "NAME"
    config.dimensions = 2
    config.states = STATES
    # -------------------------------------------------------------------------

    # ---- Override the defaults below (these may be changed at anytime) ----

    # config.state_colors = [(0,0,0),(1,1,1)]
    # config.grid_dims = (200,200)

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
    # YOUR CODE HERE
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
