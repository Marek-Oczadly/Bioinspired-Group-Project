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
from ca_utils import (MOORE_OFFSETS, ft_per_min_to_kmh, kmh_to_ft_per_min,
                      neighbour_distances, values_at_neighbours,
                      wind_components)
from place_forest import place_forest
from place_water import place_water
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

CELL_SIZE_KM = CONFIG["cell_size_km"]
WIND_FOCUS_PER_KMH = CONFIG["wind_focus_per_kmh"]
BURNT_DARKNESS = CONFIG["burnt_darkness"]
BURNING_TOWN_DARKNESS = CONFIG["burning_town_darkness"]
# How long water keeps a cell wet (and so unable to burn)
TIME_TO_DRY_HOURS = CONFIG["time_to_dry_hours"]
WET_COLOUR = tuple(CONFIG["wet_colour"])
# How much of a wet cell's colour is the wet colour (0 = none, 1 = all)
WET_BLEND = CONFIG["wet_blend"]


FALLBACK_HOURS_PER_STEP = 1.0
# Float slack so a front that should arrive exactly this step isn't a hair short
PROGRESS_TOLERANCE = 1e-9
# The same for times, so a cell due to dry exactly now isn't a step late
HOURS_TOLERANCE = 1e-9

DIRECTION_NAMES = ("NW", "N", "NE", "W", "E", "SW", "S", "SE")


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


class Wetness(Enum):
    """Marks a display key as the wet version of another one."""
    WET = auto()


@dataclass
class RunConstants:
    """The settings from run_constants.json that change between runs.

    wind_direction: where the wind comes FROM, in degrees clockwise from
        north (0 = northerly, 90 = easterly, ...).
    wind_speed: km/h.
    spread_randomness: how much each fire front's speed varies at random, as
        a fraction of its Rothermel speed (0 = none, 0.3 = about 30%). The
        average speed is unchanged.
    random_seed: a whole number to repeat a run exactly, or null for a
        different run each time.
    time_to_respond_hours: how long after the start the water arrives. The
        step length is worked out from the wind, so this is rounded to the
        nearest whole step.
    water_area_km2: how much water there is, as the area it covers.
    """
    burn_incinerator: bool
    burn_power_plant: bool
    wind_direction: float
    wind_speed: float
    spread_randomness: float = 0.0
    random_seed: Optional[int] = None
    time_to_respond_hours: float = 12.0
    water_area_km2: float = 12.5

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


def blacken(colour, darkness=BURNT_DARKNESS):
    """The colour scaled towards black (darkness is the brightness kept)."""
    return tuple(channel * darkness for channel in colour)


BURNT_COLOURS = {terrain: blacken(colour)
                 for terrain, colour in TERRAIN_COLOURS.items()}

# The town stays recognisable once on fire, as a darkened grey, instead of
# showing the chaparral underneath it. Burning is less dark than burnt out.
TOWN_BURNING_COLOUR = blacken(SPECIAL_COLOURS[SpecialType.TOWN],
                              BURNING_TOWN_DARKNESS)
TOWN_BURNT_COLOUR = blacken(SPECIAL_COLOURS[SpecialType.TOWN])

# Hours a cell burns before burning out
BURN_DURATION_HOURS = terrain_setting("burn_duration_hours")

TERRAIN_DEFAULTS = {
    terrain: dict(
        colour=TERRAIN_COLOURS[terrain],
        max_burn_duration=BURN_DURATION_HOURS[terrain],
        water_fraction=0.0,
    )
    for terrain in Terrain
}


@dataclass
class Cell:
    terrain: Terrain
    colour: Tuple[float, float, float]  # RGB, each 0-1
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
# (Terrain, BURNING) or (Terrain, BURNT), and the same two for the town
DISPLAY_COLOURS = {}
DISPLAY_COLOURS.update(TERRAIN_COLOURS)
DISPLAY_COLOURS.update(SPECIAL_COLOURS)
DISPLAY_COLOURS.update({(terrain, BurnState.BURNING): colour
                        for terrain, colour in BURNING_COLOURS.items()})
DISPLAY_COLOURS.update({(terrain, BurnState.BURNT): colour
                        for terrain, colour in BURNT_COLOURS.items()})
DISPLAY_COLOURS[(SpecialType.TOWN, BurnState.BURNING)] = TOWN_BURNING_COLOUR
DISPLAY_COLOURS[(SpecialType.TOWN, BurnState.BURNT)] = TOWN_BURNT_COLOUR


def with_blue_hue(colour):
    """The colour with the wet colour blended in."""
    return tuple((1 - WET_BLEND) * channel + WET_BLEND * wet_channel
                 for channel, wet_channel in zip(colour, WET_COLOUR))


def can_be_wet(key):
    """Whether a display key has a wet version. Water puts out fire, so a
    burning cell is never wet, and a lake ignores water."""
    if isinstance(key, tuple):
        owner, burn_state = key
        return burn_state != BurnState.BURNING and owner != Terrain.LAKE
    return key != Terrain.LAKE


# Each wet version is the dry colour with a blue hue. Wet keys are written
# (Wetness.WET, dry key) and come after every dry key.
for dry_key in list(DISPLAY_COLOURS):
    if can_be_wet(dry_key):
        DISPLAY_COLOURS[(Wetness.WET, dry_key)] = with_blue_hue(
            DISPLAY_COLOURS[dry_key])

DISPLAY_KEYS = list(DISPLAY_COLOURS)
STATES = list(range(len(DISPLAY_KEYS)))
STATE_COLOURS = [DISPLAY_COLOURS[key] for key in DISPLAY_KEYS]

# Lookups from one grid state number to the other (-1 where there is none)
WET_STATE = np.full(len(DISPLAY_KEYS), -1)
DRY_STATE = np.full(len(DISPLAY_KEYS), -1)
for wet_number, wet_key in enumerate(DISPLAY_KEYS):
    if isinstance(wet_key, tuple) and wet_key[0] is Wetness.WET:
        dry_number = DISPLAY_KEYS.index(wet_key[1])
        WET_STATE[dry_number] = wet_number
        DRY_STATE[wet_number] = dry_number

# Every grid state that means "on fire", which the transition function needs
BURNING_STATE_NUMBERS = [
    DISPLAY_KEYS.index((owner, BurnState.BURNING))
    for owner in [*Terrain, SpecialType.TOWN]]


def display_state(cell):
    """The numeric grid state of a cell. An unburnt special type hides its
    terrain, and a burning or burnt town keeps its own look; other cells show
    their terrain burning or burnt."""
    if cell.burn_state != BurnState.UNBURNT:
        owner = (SpecialType.TOWN if cell.special_type == SpecialType.TOWN
                 else cell.terrain)
        key = (owner, cell.burn_state)
    elif cell.special_type is not None:
        key = cell.special_type
    else:
        key = cell.terrain
    return DISPLAY_KEYS.index(key)


def stage_to_grid(stage):
    return np.array([[display_state(cell) for cell in row] for row in stage])


def random_streams(run):
    """Independent random seeds for the spread speeds, the water and the
    forest, all from run.random_seed, so a seeded run repeats exactly and
    none of the three changes another."""
    spread, water, forest = np.random.SeedSequence(run.random_seed).spawn(3)
    return spread, water, forest


def plant_forest(stage, rng):
    """Turn the cells place_forest picks into dense forest.

    place_forest only decides where, so what can't take forest is enforced
    here: the lake is skipped. A town, incinerator or power plant keeps its
    special type and just gets forest fuel underneath.
    """
    for row, col in place_forest(stage, rng):
        cell = stage[row][col]
        if cell.terrain == Terrain.LAKE:
            continue
        stage[row][col] = Cell.from_terrain(
            Terrain.DENSE_FOREST, special_type=cell.special_type)


def build_stage(run):
    """The stage for a run: the default stage with the extra forest planted,
    then the ignition points lit. The forest is random, so call this once and
    use the result for everything in the run."""
    stage = load_stage(STAGE_PATH)
    forest_rng = np.random.default_rng(random_streams(run)[2])
    plant_forest(stage, forest_rng)
    begin_ignition(stage, run.ignition_points)
    return stage


@dataclass(frozen=True)
class FuelModel:
    """The Rothermel (1972) inputs for one terrain, from Anderson's fuel
    models. Rothermel works in ft/min; see "Statistics Needed for Model
    Design.doc" on the drive for where the values come from."""
    heat_released: float
    share_reaching_fuel: float  # share of the heat reaching unburnt fuel
    heat_to_ignite: float
    wind_constant: float  # a
    wind_power: float  # b
    packing_term: float  # p
    wind_factor: float  # share of the open wind reaching the flames
    slope_constant: float  # k
    slope_degrees: float

    @property
    def no_wind_rate(self):
        """R0: the spread rate on flat ground with no wind, in ft/min."""
        return (self.heat_released * self.share_reaching_fuel
                / self.heat_to_ignite)

    @property
    def slope_boost(self):
        """S = k * tan(slope)^2"""
        return (self.slope_constant
                * math.tan(math.radians(self.slope_degrees)) ** 2)


FUEL_MODELS = {terrain: FuelModel(**fuel)
               for terrain, fuel in terrain_setting("fuel").items()}


def rothermel_rate_kmh(fuel, wind_kmh, alignment=1.0):
    """Fire spread rate (km/h) through a fuel: R = R0 * (1 + W + S).

    alignment: cosine between the wind and the fire's direction of travel
        (1 = head fire, running with the wind). The wind boost W is scaled by
        alignment raised to 1 + WIND_FOCUS_PER_KMH * wind speed, so it fades
        off-axis (faster in a stronger wind) and is 0 across or against the
        wind. Flank and backing fires then spread at the no-wind rate.

    W = a * U^b * p, where U is the open wind (ft/min) * wind factor.
    """
    midflame_wind = kmh_to_ft_per_min(wind_kmh) * fuel.wind_factor
    wind_boost = (fuel.wind_constant * midflame_wind ** fuel.wind_power
                  * fuel.packing_term)
    focus_power = 1 + WIND_FOCUS_PER_KMH * wind_kmh
    wind_boost *= max(0.0, alignment) ** focus_power
    return ft_per_min_to_kmh(
        fuel.no_wind_rate * (1 + wind_boost + fuel.slope_boost))


def travel_alignments(east, north, offsets=MOORE_OFFSETS):
    """Cosine between the wind and the fire's direction of travel into a cell
    from each neighbour, in the order of offsets.

    east, north: the wind components from wind_components().
    """
    alignments = []
    for offset_east, offset_north in offsets:
        length = math.hypot(offset_east, offset_north)
        # Fire travels from the neighbour into the cell, opposite the offset
        travel_east = -offset_east / length
        travel_north = -offset_north / length
        alignments.append(
            round(travel_east * east + travel_north * north, 10))
    return alignments


def harmonic_mean(first, second):
    """Element-wise harmonic mean, 0 wherever either value is 0."""
    total = first + second
    return np.divide(2 * first * second, total,
                     out=np.zeros_like(total), where=total > 0)


def directional_rates(stage, run):
    """How fast fire spreads into each cell from each neighbour.

    Returns one array per neighbour (in the order of MOORE_OFFSETS) holding,
    for every cell, the fraction of the gap to that neighbour crossed per hour
    when the neighbour is burning.

    A fire crossing between two cells spends half the gap in each, so its
    speed is the harmonic mean of the two terrains' rates. Inside one terrain
    that is just the terrain's rate; against a lake it is 0.
    """
    wind_east, wind_north = wind_components(run.wind_direction)
    alignments = travel_alignments(wind_east, wind_north)
    distances = neighbour_distances(CELL_SIZE_KM)

    rates = []
    for offset, alignment, distance in zip(MOORE_OFFSETS, alignments,
                                           distances):
        speed_by_terrain = {
            terrain: rothermel_rate_kmh(
                FUEL_MODELS[terrain], run.wind_speed, alignment)
            for terrain in Terrain}
        own_speed = np.array([[speed_by_terrain[cell.terrain] for cell in row]
                              for row in stage])
        neighbour_speed = values_at_neighbours(own_speed, (offset,))[0]
        rates.append(harmonic_mean(own_speed, neighbour_speed) / distance)
    return rates


def spread_noise(shape, randomness, seed=None):
    """A random speed multiplier for every cell and neighbour direction.

    Lognormal, so it is never negative, and shifted to have a mean of 1 so the
    average spread rate is still the Rothermel rate. It is drawn once per run
    (not every step), so how random the fire is doesn't depend on the time
    per step. randomness is the spread of the log of the multiplier; 0 gives
    exactly 1 everywhere.
    """
    if randomness <= 0:
        return np.ones(shape)
    return np.random.default_rng(seed).lognormal(
        mean=-randomness ** 2 / 2, sigma=randomness, size=shape)


def format_duration(hours):
    return "{:.1f} min".format(hours * 60) if hours < 1 \
        else "{:.2f} h".format(hours)


@dataclass(frozen=True)
class TimeStep:
    hours: float
    reason: str  # what limits it, for display


def choose_time_step(rates, stage):
    """The time per generation: how long the fastest fire front takes to
    cross exactly one gap between neighbouring cells.

    The CA can only move fire one cell per step, so any longer step would
    slow the fastest fire down. Slower fronts need several steps per gap.
    """
    fastest = max(range(len(rates)), key=lambda d: rates[d].max())
    fastest_rate = rates[fastest].max()
    if fastest_rate <= 0:
        return TimeStep(FALLBACK_HOURS_PER_STEP, "no fire can spread")

    row, col = np.unravel_index(rates[fastest].argmax(),
                                rates[fastest].shape)
    distance = neighbour_distances(CELL_SIZE_KM)[fastest]
    reason = "{} spreading in from the {} at {:.3f} km/h over {:.2f} km".format(
        stage[row][col].terrain.name, DIRECTION_NAMES[fastest],
        fastest_rate * distance, distance)
    return TimeStep(1 / fastest_rate, reason)


def response_step(run, time_step):
    """How many steps have passed when the water arrives: the response time
    rounded to the nearest whole step (the step length depends on the wind)."""
    return max(0, round(run.time_to_respond_hours / time_step.hours))


def water_squares(run):
    """How many cells the water covers: its area divided by the area of one
    cell (0.5 km by 0.5 km is 0.25 km^2). place_water is only told this."""
    return int(math.floor(run.water_area_km2 / CELL_SIZE_KM ** 2 + 1e-9))


def describe_run(run, time_step):
    """Text for the console: the wind, the spread speeds, the time step and
    when the water arrives."""
    lines = ["Wind {:.1f} km/h from {:.0f} degrees".format(
        run.wind_speed, run.wind_direction)]
    for terrain in Terrain:
        fuel = FUEL_MODELS[terrain]
        lines.append(
            "  {:<17} head fire {:.3f} km/h, no wind {:.3f} km/h, "
            "burns {} h".format(
                terrain.name,
                rothermel_rate_kmh(fuel, run.wind_speed),
                rothermel_rate_kmh(fuel, 0.0),
                BURN_DURATION_HOURS[terrain]))
    lines.append("Time per step: {} ({:.4f} h), set by {}".format(
        format_duration(time_step.hours), time_step.hours, time_step.reason))
    lines.append("Steps per day: {:.0f}".format(24 / time_step.hours))
    lines.append("Spread randomness: {:.2f} ({})".format(
        run.spread_randomness,
        "seed {}".format(run.random_seed) if run.random_seed is not None
        else "different every run"))
    water_step = response_step(run, time_step)
    lines.append(
        "Water: {:.1f} km2 = {} cells, arrives after step {} ({:.1f} h, "
        "asked for {:g} h), each wet cell dries after {:g} h".format(
            run.water_area_km2, water_squares(run), water_step,
            water_step * time_step.hours, run.time_to_respond_hours,
            TIME_TO_DRY_HOURS))
    return "\n".join(lines)


@dataclass
class FireData:
    """Per-cell data (arrays shaped like the grid) that the grid's single
    state number can't hold. The transition function updates front_progress,
    caught_fire, burn_time, head_start, wet, wet_until, steps_taken and
    water_placed."""
    rates: np.ndarray  # from directional_rates(), indexed [neighbour, row, col]
    time_step: TimeStep
    front_progress: np.ndarray  # 0-1 per neighbour: how far its front has got
    caught_fire: np.ndarray  # bool, True once ignited (or burning from the start)
    burn_duration: np.ndarray  # hours
    burn_time: np.ndarray  # hours burnt so far
    head_start: np.ndarray  # hours burnt that the neighbours haven't counted
    burning_state: np.ndarray  # grid state when burning
    burnt_state: np.ndarray  # grid state when burnt out

    town_mask: np.ndarray  # bool, which cells are the town
    lake_mask: np.ndarray  # bool, which cells are lake (water does nothing)
    wet: np.ndarray  # bool, cells currently wet: they can't catch fire
    wet_until: np.ndarray  # hour at which each wet cell dries
    response_step: int  # steps taken when the water arrives
    water_cells: int  # how many cells the water covers, worked out once
    water_rng: np.random.Generator  # for place_water, so runs can repeat

    town_reached: bool = False  # True once any town cell has caught fire
    steps_taken: int = 0
    water_placed: bool = False  # the water arrives once

    @property
    def hours_per_step(self):
        return self.time_step.hours

    @classmethod
    def from_stage(cls, stage, run):
        def per_cell(get_value):
            return np.array([[get_value(cell) for cell in row]
                             for row in stage], dtype=float)

        def state_when(burn_state):
            return per_cell(lambda cell: display_state(
                replace(cell, burn_state=burn_state)))

        rates = np.array(directional_rates(stage, run))
        # The step comes from the noise-free speeds, so it stays the time the
        # fastest typical front takes to cross a cell. A front that happens to
        # draw a big multiplier still crosses at most one cell per step.
        time_step = choose_time_step(rates, stage)
        noise_seed, water_seed, _ = random_streams(run)
        rates *= spread_noise(rates.shape, run.spread_randomness, noise_seed)
        caught_fire = per_cell(
            lambda cell: cell.burn_state != BurnState.UNBURNT).astype(bool)
        town_mask = per_cell(
            lambda cell: cell.special_type == SpecialType.TOWN).astype(bool)
        lake_mask = per_cell(
            lambda cell: cell.terrain == Terrain.LAKE).astype(bool)
        grid_shape = (len(stage), len(stage[0]))
        return cls(
            rates=rates,
            time_step=time_step,
            front_progress=np.zeros(rates.shape),
            caught_fire=caught_fire,
            town_mask=town_mask,
            lake_mask=lake_mask,
            wet=np.zeros(grid_shape, dtype=bool),
            wet_until=np.zeros(grid_shape),
            response_step=response_step(run, time_step),
            water_cells=water_squares(run),
            water_rng=np.random.default_rng(water_seed),
            town_reached=bool(np.any(town_mask & caught_fire)),
            burn_duration=per_cell(lambda cell: cell.max_burn_duration),
            burn_time=np.zeros(grid_shape),
            head_start=np.zeros(grid_shape),
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
    stage = build_stage(run)
    initial_grid = stage_to_grid(stage)
    config.grid_dims = initial_grid.shape
    config.initial_grid = initial_grid

    # The time per step comes from the spread speeds, so tell the user what
    # one generation means (the GUI shows what a description prints)
    fire = FireData.from_stage(stage, run)
    print(describe_run(run, fire.time_step))

    # ----------------------------------------------------------------------

    # the GUI calls this to pass the user defined config
    # into the main system with an extra argument
    # do not change
    if len(args) == 2:
        config.save()
        sys.exit()
    return config


def dry_out(grid, fire, now):
    """Let wet cells whose time is up go back to how they were."""
    drying = fire.wet & (fire.wet_until <= now + HOURS_TOLERANCE)
    grid[drying] = DRY_STATE[grid[drying].astype(int)]
    fire.wet &= ~drying


def drop_water(grid, fire, now):
    """Ask place_water where to put the water, then wet those cells.

    place_water only decides where, so how much is used is enforced here: if
    it returns more than fire.water_cells cells, only the first ones are used.
    Lake cells are ignored. A wet cell can't catch fire for TIME_TO_DRY_HOURS,
    and a cell that was burning is put out: it stays burnt, since the fuel it
    burned is gone.
    """
    cells = place_water(grid.copy(), fire.water_cells, fire.water_rng)
    cells = cells[:fire.water_cells]
    fire.water_placed = True

    hit = np.zeros(grid.shape, dtype=bool)
    hit[cells[:, 0], cells[:, 1]] = True
    hit &= ~fire.lake_mask

    extinguished = hit & np.isin(grid, BURNING_STATE_NUMBERS)
    grid[extinguished] = fire.burnt_state[extinguished]
    fire.head_start[extinguished] = 0.0

    # Fronts that were heading for these cells start again once they dry
    fire.front_progress[:, hit] = 0.0
    fire.wet |= hit
    fire.wet_until[hit] = now + TIME_TO_DRY_HOURS
    grid[hit] = WET_STATE[grid[hit].astype(int)]


def transition_function(grid, neighbourstates, neighbourcounts, fire):
    """Apply one step of fire to the grid and return the new grid. There is
    no randomness: the same run always burns the same way.

    fire: per-cell burn data and spread rates, updated in place. One step is
        fire.hours_per_step hours.

    0. Wet cells that have dried go back to normal, and once, after
       fire.response_step steps, the water arrives (see drop_water).
    1. Each burning neighbour is a separate fire front moving into a cell,
       and each front keeps its own progress (the fraction of the gap to the
       cell it has crossed). Progress is kept if the neighbour burns out,
       since the front stays where it got to.
    2. A cell catches fire when its first front crosses the gap (progress 1).
       Fronts don't add up: a second neighbour doesn't speed the first.
    3. Burning cells age and burn out after their terrain's burn duration.

    A front rarely arrives exactly on a step boundary. A cell reached part-way
    through a step really caught fire that many hours before the step ended,
    so it starts with that much burn time and its neighbours are credited
    with it (its head_start) on the next step. Without this every cell would
    round up to a whole number of steps, and slow fronts (which need several
    steps per cell) would be noticeably slower than their spread rate.
    """
    # 0. Water. Cells put out by the water are no longer burning below.
    now = fire.steps_taken * fire.hours_per_step
    dry_out(grid, fire, now)
    if not fire.water_placed and fire.steps_taken >= fire.response_step:
        drop_water(grid, fire, now)

    burning = np.isin(grid, BURNING_STATE_NUMBERS)
    # Wet cells can't catch fire, and fronts don't advance into them
    waiting = ~fire.caught_fire & ~fire.wet

    # 1. Advance the front from each burning neighbour
    burning_hours = fire.hours_per_step + fire.head_start
    for front_progress, rate, neighbour_burning, neighbour_hours in zip(
            fire.front_progress, fire.rates,
            values_at_neighbours(burning),
            values_at_neighbours(burning_hours)):
        front_progress += np.where(neighbour_burning & waiting,
                                   rate * neighbour_hours, 0.0)

    # 2. A cell ignites when any front has crossed
    arrived = fire.front_progress >= 1.0 - PROGRESS_TOLERANCE
    ignited = waiting & arrived.any(axis=0)
    # How long before the end of the step the first front got there
    hours_past_arrival = np.divide(
        fire.front_progress - 1.0, fire.rates,
        out=np.zeros(fire.rates.shape), where=fire.rates > 0)
    surplus_hours = np.where(arrived, hours_past_arrival, 0.0).max(axis=0)
    surplus_hours = np.clip(surplus_hours, 0.0, fire.hours_per_step)
    fire.caught_fire |= ignited

    # 3. Age burning cells and burn out the finished ones
    fire.burn_time[burning] += fire.hours_per_step
    fire.head_start[burning] = 0.0
    burnt_out = burning & (fire.burn_time >= fire.burn_duration)

    fire.burn_time[ignited] = surplus_hours[ignited]
    fire.head_start[ignited] = surplus_hours[ignited]
    grid[burnt_out] = fire.burnt_state[burnt_out]
    grid[ignited] = fire.burning_state[ignited]

    # 4. Update whether the fire has reached any town cells
    fire.town_reached = fire.town_reached or bool(np.any(ignited & fire.town_mask))

    fire.steps_taken += 1
    return grid


def main():
    config = setup(sys.argv[1:])
    run = load_run_constants()

    # The forest is random, so setup() built a different stage. The stage that
    # the fire runs on must also be the one that is drawn.
    stage = build_stage(run)
    config.initial_grid = stage_to_grid(stage)
    fire = FireData.from_stage(stage, run)

    grid = Grid2D(config, (transition_function, fire))
    timeline = grid.run()

    config.save()
    utils.save(timeline, config.timeline_path)

if __name__ == "__main__":
    main()
