"""Run CaPyle simulations from the terminal."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import repeat
import json
from pathlib import Path
import sys

# CaPyle import fix. ca_descriptions is on the path too, as it is when the GUI
# runs a description as a script, so forest_fires can import its ca_utils.
ROOT = Path(__file__).resolve().parent
for folder in (ROOT, ROOT / 'capyle', ROOT / 'capyle/ca',
               ROOT / 'capyle/guicomponents', ROOT / 'ca_descriptions'):
    sys.path.append(str(folder))

from capyle.ca import CAConfig, Grid2D

from ca_descriptions.forest_fires import (
    STATES, FireData,
    build_stage, load_run_constants, stage_to_grid, transition_function,
)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return number


def main():
    # command line parsing
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'simulation_config.json',
                        help='JSON settings file (default: simulation_config.json)')
    parser.add_argument('--runs', type=positive_int, help='override the configured run count')
    parser.add_argument('--generations', type=positive_int, help='override the configured generation count')
    parser.add_argument('--workers', type=positive_int, help='override the configured worker count')
    args = parser.parse_args()

    # error handling
    try:
        settings = json.loads(args.config.read_text())
        runs = args.runs if args.runs is not None else settings['runs']
        generations = args.generations if args.generations is not None else settings['generations']
        workers = args.workers if args.workers is not None else settings['workers']
        grid_size = settings['grid_size']
        seed = settings['seed']
        for name, value in [('runs', runs), ('generations', generations), ('workers', workers)]:
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not isinstance(grid_size, list) or len(grid_size) != 2 or any(
            type(size) is not int or size < 3 for size in grid_size
        ):
            raise ValueError('grid_size must contain two integers of at least 3')
        if type(seed) is not int or seed < 0:
            raise ValueError('seed must be a non-negative integer')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f'{args.config}: {error}')

    if workers == 1 or runs == 1:
        for run in range(runs):
            arrival_hours = run_simulation(run, generations, grid_size, seed)
            print_result(run + 1, arrival_hours, generations)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, runs)) as pool:
            arrival_times = pool.map(run_simulation, range(runs), repeat(generations),
                                     repeat(grid_size), repeat(seed))
            for run_number, arrival_hours in enumerate(arrival_times, start=1):
                print_result(run_number, arrival_hours, generations)


def print_result(run_number, arrival_hours, generations):
    if arrival_hours is None:
        print(f'Run {run_number}: Fire did not reach town within {generations} generations.')
    else:
        print(f'Run {run_number}: Fire reached town after {arrival_hours:g} simulated hours.')


def run_simulation(run, generations, grid_size, seed):
    """Return hours to first town ignition, or None if the run reaches its limit."""
    # The forest-fire model draws its randomness from random_seed, so a
    # different seed per run gives different runs that repeat exactly
    run_constants = replace(load_run_constants(), random_seed=seed + run)
    stage = build_stage(run_constants.ignition_points)
    initial_grid = stage_to_grid(stage)
    if tuple(grid_size) != initial_grid.shape:
        raise ValueError(f'Configured grid_size must match the stage: {initial_grid.shape}')

    config = CAConfig(str(ROOT / 'ca_descriptions/forest_fires.py'))
    config.states = STATES
    config.grid_dims = initial_grid.shape
    config.num_generations = generations
    config.initial_grid = initial_grid
    config.wrap = False
    config.fill_in_defaults()

    fire = FireData.from_stage(stage, run_constants)
    if not fire.town_mask.any():
        raise ValueError('The stage has no town cells')
    simulation = Grid2D(config, (transition_function, fire))

    step = 0
    while step < generations and not fire.town_reached:
        simulation.step()
        step += 1

    # The hours per step depend on the wind, so they come from the fire data
    return step * fire.hours_per_step if fire.town_reached else None


if __name__ == '__main__':
    main()
