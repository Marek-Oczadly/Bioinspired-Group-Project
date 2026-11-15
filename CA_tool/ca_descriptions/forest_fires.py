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

import json
import math
import os
import numpy as np
from capyle.ca import Grid2D, Neighbourhood, randomise2d
import capyle.utils as utils
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Optional, Tuple


META_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "forest_fires_meta")


def load_json(file_name):
    with open(os.path.join(META_DIR, file_name), "r") as f:
        return json.load(f)

CONFIG = load_json("config.json")

STAGE_PATH = os.path.join(META_DIR, CONFIG["stage_file"])

HOURS_PER_STEP = CONFIG["hours_per_step"]
PEAK_SPREAD_CHANCE_PER_HOUR = CONFIG["peak_spread_chance_per_hour"]
WIND_SPREAD_FRACTION = CONFIG["wind_spread_fraction"]
WIND_FOCUS_PER_KMH = CONFIG["wind_focus_per_kmh"]
NEUTRAL_WIND_MULTIPLIER = CONFIG["neutral_wind_multiplier"]
BURNT_DARKNESS = CONFIG["burnt_darkness"]


def wind_components(direction):
    """Split a wind direction (degrees, see RunConstants) into east and north
    components, each -1 to 1, for the way the wind is BLOWING TOWARDS."""
    radians = math.radians(direction % 360)
    east = -math.sin(radians)
    north = -math.cos(radians)
    return east, north


# (NW, N, NE, W, E, SW, S, SE)
MOORE_OFFSETS = ((-1, 1), (0, 1), (1, 1),
                     (-1, 0), (1, 0),
                     (-1, -1), (0, -1), (1, -1))


def wind_multipliers(east, north, speed, offsets=MOORE_OFFSETS):
    """How much the wind boosts fire spread into a cell from each neighbour.

    east, north: the wind components from wind_components().
    speed: the wind speed in km/h.
    offsets: (east, north) offset of each neighbour; the result lines up with
        neighbourstates.

    Returns a read-only array with one multiplier per neighbour:
        multiplier = NEUTRAL_WIND_MULTIPLIER * e^effect
    so it is neutral (0.5) when the wind is absent or perpendicular, shrinks
    towards 0 upwind and grows without limit downwind.

    effect = forward spread rate * focused alignment, where the forward spread
    rate is WIND_SPREAD_FRACTION of the wind speed. The alignment is the
    cosine between the wind and the fire's direction of travel, raised
    (keeping its sign) to 1 + WIND_FOCUS_PER_KMH * speed. Stronger wind means
    a bigger power, so the effect on diagonals shrinks faster than along the
    wind axis.
    """
    forward_spread_rate = WIND_SPREAD_FRACTION * speed
    focus_power = 1 + WIND_FOCUS_PER_KMH * speed
    multipliers = []
    for offset_east, offset_north in offsets:
        length = math.hypot(offset_east, offset_north)
        # Fire travels from the neighbour into the cell, opposite the offset
        travel_east = -offset_east / length
        travel_north = -offset_north / length
        alignment = round(travel_east * east + travel_north * north, 10)
        focused_alignment = math.copysign(abs(alignment) ** focus_power,
                                          alignment)
        effect = forward_spread_rate * focused_alignment
        multipliers.append(NEUTRAL_WIND_MULTIPLIER * math.exp(effect))
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


@dataclass
class RunConstants:
    """The settings from run_constants.json that change between runs.

    wind_direction: where the wind comes FROM, in degrees clockwise from
        north (0 = northerly, 90 = easterly, ...).
    wind_speed: km/h.
    """
    burn_incinerator: bool
    burn_power_plant: bool
    wind_direction: float
    wind_speed: float

    @property
    def ignition_points(self):
        return tuple(
            point for point, burn in (
                (SpecialType.INCINERATOR, self.burn_incinerator),
                (SpecialType.POWER_PLANT, self.burn_power_plant))
            if burn)


def load_run_constants():
    return RunConstants(**load_json("run_constants.json"))


def terrain_setting(key):
    """The config.json value called `key` for each terrain."""
    return {terrain: CONFIG["terrain"][terrain.name][key]
            for terrain in Terrain}


# Colours are sampled from the reference map (forest_fires_meta/town.png).
TERRAIN_COLOURS = {terrain: tuple(colour)
                   for terrain, colour in terrain_setting("colour").items()}

SPECIAL_COLOURS = {special: tuple(CONFIG["special_colours"][special.name])
                   for special in SpecialType}

# Orange, shifted per terrain so the burning terrain is still identifiable
BURNING_COLOURS = {terrain: tuple(colour) for terrain, colour
                   in terrain_setting("burning_colour").items()}


def blacken(colour):
    return tuple(channel * BURNT_DARKNESS for channel in colour)


BURNT_COLOURS = {terrain: blacken(colour)
                 for terrain, colour in TERRAIN_COLOURS.items()}

# TODO: the burn properties in config.json are placeholders
# Chance (0-1) of catching fire, before the wind multiplier is applied
IGNITE_PROBABILITY = terrain_setting("ignite_probability")
# Hours a cell burns before burning out
BURN_DURATION_HOURS = terrain_setting("burn_duration_hours")

TERRAIN_DEFAULTS = {
    terrain: dict(
        colour=TERRAIN_COLOURS[terrain],
        burn_probability=IGNITE_PROBABILITY[terrain],
        max_burn_duration=BURN_DURATION_HOURS[terrain],
        water_fraction=0.0,
    )
    for terrain in Terrain
}


@dataclass
class Cell:
    terrain: Terrain
    colour: Tuple[float, float, float]  # RGB, each 0-1
    burn_probability: float
    remaining_burn_duration: float
    max_burn_duration: float
    special_type: Optional[SpecialType] = None
    water_fraction: float = 0.0
    burn_state: BurnState = BurnState.UNBURNT

    @classmethod
    def from_terrain(cls, terrain, **overrides):
        """Create a cell with its terrain's defaults, overridable by keyword."""
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
    """Load a stage (a list of rows of unburnt Cells) from a text file.

    Cells are space separated, rows are newline separated. Each cell is a
    terrain code (C, D, S, L) then a special code (T, I, P, N for none),
    e.g. "DN LN CT".

    Raises ValueError on an unknown code or rows of differing length.
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
    for row in stage:
        for cell in row:
            if cell.special_type in ignition_points:
                cell.burn_state = BurnState.BURNING


# One key per distinct look: Terrain or SpecialType (unburnt),
# (Terrain, BURNING) or (Terrain, BURNT)
DISPLAY_COLOURS = {}
DISPLAY_COLOURS.update(TERRAIN_COLOURS)
DISPLAY_COLOURS.update(SPECIAL_COLOURS)
DISPLAY_COLOURS.update({(terrain, BurnState.BURNING): colour
                        for terrain, colour in BURNING_COLOURS.items()})
DISPLAY_COLOURS.update({(terrain, BurnState.BURNT): colour
                        for terrain, colour in BURNT_COLOURS.items()})

DISPLAY_KEYS = list(DISPLAY_COLOURS)
STATES = list(range(len(DISPLAY_KEYS)))
STATE_COLOURS = [DISPLAY_COLOURS[key] for key in DISPLAY_KEYS]

BURNING_STATE_NUMBERS = [DISPLAY_KEYS.index((terrain, BurnState.BURNING))
                         for terrain in Terrain]


def display_state(cell):
    """The numeric grid state of a cell. An unburnt special type hides its
    terrain."""
    if cell.burn_state != BurnState.UNBURNT:
        key = (cell.terrain, cell.burn_state)
    elif cell.special_type is not None:
        key = cell.special_type
    else:
        key = cell.terrain
    return DISPLAY_KEYS.index(key)


def stage_to_grid(stage):
    return np.array([[display_state(cell) for cell in row] for row in stage])


def build_stage(ignition_points):
    stage = load_stage(STAGE_PATH)
    begin_ignition(stage, ignition_points)
    return stage


@dataclass
class FireData:
    """Per-cell data (arrays shaped like the grid) that the grid's single
    state number can't hold. The transition function updates burn_time and
    ignite_probability."""
    ignite_probability: np.ndarray  # 0 once burnt out
    burn_duration: np.ndarray  # hours
    burn_time: np.ndarray  # hours burnt so far
    burning_state: np.ndarray  # grid state when burning
    burnt_state: np.ndarray  # grid state when burnt out

    @classmethod
    def from_stage(cls, stage):
        def per_cell(get_value):
            return np.array([[get_value(cell) for cell in row]
                             for row in stage], dtype=float)

        def state_when(burn_state):
            return per_cell(lambda cell: display_state(
                replace(cell, burn_state=burn_state)))

        return cls(
            ignite_probability=per_cell(lambda cell: cell.burn_probability),
            burn_duration=per_cell(lambda cell: cell.max_burn_duration),
            burn_time=np.zeros((len(stage), len(stage[0]))),
            burning_state=state_when(BurnState.BURNING),
            burnt_state=state_when(BurnState.BURNT),
        )


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
    # The stage has real edges, so fire must not wrap around
    config.wrap = False
    run = load_run_constants()
    initial_grid = stage_to_grid(build_stage(run.ignition_points))
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


def spread_chance_per_hour(fraction_burnt):
    """Chance per hour that a burning cell spreads, given how far through its
    burn time it is (0 = just ignited, 1 = burnt out).

    Starts at 0, peaks at PEAK_SPREAD_CHANCE_PER_HOUR two-thirds of the way
    through, and returns to 0. x^2 (1 - x) peaks at 4/27, hence the division.
    """
    x = np.clip(fraction_burnt, 0.0, 1.0)
    return PEAK_SPREAD_CHANCE_PER_HOUR * x ** 2 * (1 - x) / (4 / 27)


def values_at_neighbours(array, offsets=MOORE_OFFSETS):
    """For each offset, an array whose value at a cell is `array` at that
    neighbour (False beyond the grid edge), in the order of offsets."""
    rows, cols = array.shape
    padded = np.pad(array, 1, constant_values=False)
    return [padded[1 - north:1 - north + rows, 1 + east:1 + east + cols]
            for east, north in offsets]


def transition_function(grid, neighbourstates, neighbourcounts,
                        multipliers, fire):
    """Apply one step of fire to the grid and return the new grid.

    multipliers: from wind_multipliers(), one per neighbour in the order of
        neighbourstates.
    fire: per-cell burn data, updated in place.

    1. Each burning cell may spread, based on its progress through its burn.
    2. A cell catches fire from each spreading neighbour with probability
       wind multiplier * ignite probability (capped at 1).
    3. Cells that finish burning burn out and can no longer ignite.
    """
    burning = np.isin(grid, BURNING_STATE_NUMBERS)

    # 1. One roll per burning cell
    fraction_burnt = fire.burn_time / np.maximum(fire.burn_duration, 1e-9)
    spread_chance = spread_chance_per_hour(fraction_burnt) * HOURS_PER_STEP
    spreading = burning & (np.random.random(grid.shape) < spread_chance)

    # 2. An independent roll per spreading neighbour
    ignited = np.zeros(grid.shape, dtype=bool)
    for neighbour_spreading, multiplier in zip(
            values_at_neighbours(spreading), multipliers):
        catch_chance = np.minimum(1.0, multiplier * fire.ignite_probability)
        ignited |= neighbour_spreading & (
            np.random.random(grid.shape) < catch_chance)
    ignited &= ~burning

    # 3. Age burning cells and burn out the finished ones
    fire.burn_time[burning] += HOURS_PER_STEP
    burnt_out = burning & (fire.burn_time >= fire.burn_duration)
    fire.ignite_probability[burnt_out] = 0.0

    grid[burnt_out] = fire.burnt_state[burnt_out]
    grid[ignited] = fire.burning_state[ignited]
    return grid


def main():
    config = setup(sys.argv[1:])
    run = load_run_constants()

    # The wind is fixed for the run, so compute the multipliers once
    multipliers = wind_multipliers(*wind_components(run.wind_direction),
                                   run.wind_speed)
    fire = FireData.from_stage(build_stage(run.ignition_points))

    grid = Grid2D(config, (transition_function, multipliers, fire))
    timeline = grid.run()

    config.save()
    utils.save(timeline, config.timeline_path)

if __name__ == "__main__":
    main()
