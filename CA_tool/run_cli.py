"""Run CaPyle simulations from the terminal."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import repeat
import json
from pathlib import Path
import sys

import numpy as np

sys.path.append('ca_descriptions')

from ca_descriptions.forest_fires import (
    BURNING_STATE_NUMBERS, STATES, FireData,
    build_stage, load_run_constants, stage_to_grid, transition_function,
)
from capyle.ca import CAConfig, Grid2D


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return number


def load_batch_settings(args): # Added error handlers for the config
    """Read the batch config, apply command line overrides, and handle all the possible errors."""
    settings = json.loads(args.config.read_text())
    if not isinstance(settings, dict):
        raise ValueError('expected a JSON object')

    # Check the final values so command line arguments can replace missing or invalid settings
    for name in ('runs', 'generations', 'workers'):
        override = getattr(args, name)
        if override is not None:
            settings[name] = override
        value = settings[name]

        # Fix because bools count as ints for some reason
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')

    # Grid size checks
    grid_size = settings['grid_size']
    if not isinstance(grid_size, list) or len(grid_size) != 2 or any(
        type(size) is not int or size < 3 for size in grid_size
    ):
        raise ValueError('grid_size must contain two integers of at least 3')

    # Seed checks
    seed = settings['seed']
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a non-negative integer')

    return settings


def main():
    """Read the settings and run the requested simulations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('simulation_config.json'), help='JSON settings file')
    parser.add_argument('--runs', type=positive_int, help='override the configured run count')
    parser.add_argument('--generations', type=positive_int, help='override the configured generation count')
    parser.add_argument('--workers', type=positive_int, help='override the configured worker count')
    args = parser.parse_args()

    # Show the config path with the error before starting any runs.
    try:
        settings = load_batch_settings(args)
    except KeyError as error:
        parser.error(f'{args.config}: missing setting {error.args[0]!r}')
    except (OSError, ValueError) as error:
        parser.error(f'{args.config}: {error}')

    runs        = settings['runs']
    generations = settings['generations']
    workers     = settings['workers']
    grid_size   = settings['grid_size']
    seed        = settings['seed']

    if workers == 1 or runs == 1:
        for run in range(runs):
            arrival_hours, elapsed_hours = simulate(run, generations, grid_size, seed)
            print_result(run + 1, arrival_hours, elapsed_hours)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, runs)) as pool:
            # Print in run order even if the workers finish out of order
            results = pool.map(simulate, range(runs), repeat(generations),
                               repeat(grid_size), repeat(seed))
            for run_number, (arrival_hours, elapsed_hours) in enumerate(results, start=1):
                print_result(run_number, arrival_hours, elapsed_hours)


def print_result(run_number, arrival_hours, elapsed_hours):
    if arrival_hours is None:
        print(f'Run {run_number}: Fire did not reach town. '
              f'Run ended after {elapsed_hours:g} simulated hours.')
    else:
        print(f'Run {run_number}: Fire reached town after {arrival_hours:g} simulated hours.')


def run_simulation(run, generations, grid_size, seed):
    """Return simulated hours to first town ignition, or None if unreached."""
    arrival_hours, _ = simulate(run, generations, grid_size, seed)
    return arrival_hours


def simulate(run, generations, grid_size, seed):
    """Run one simulation and return arrival time and elapsed time in sim hours."""
    # Each run gets a repeatable seed
    run_constants = replace(load_run_constants(), random_seed=seed + run)
    stage = build_stage(run_constants)
    initial_grid = stage_to_grid(stage)

    # Config for CA
    config                 = CAConfig('ca_descriptions/forest_fires.py')
    config.states          = STATES
    config.grid_dims       = initial_grid.shape
    config.num_generations = generations
    config.initial_grid    = initial_grid
    config.wrap            = False  # Opposite map edge wrapping
    config.fill_in_defaults()

    fire = FireData.from_stage(stage, run_constants)
    simulation = Grid2D(config, (transition_function, fire))

    step = 0
    while step < generations and not fire.town_reached:
        # An extinguished fire can't start spreading again
        if not np.isin(simulation.grid, BURNING_STATE_NUMBERS).any():
            break

        simulation.step()
        step += 1

    elapsed_hours = step * fire.hours_per_step
    if not fire.town_reached:
        return None, elapsed_hours

    # Town cells can ignite partway through the final step so the town cell
    # that has burned the longest marks the actual arrival.
    time_since_arrival = float(fire.burn_time[fire.town_mask].max())
    return max(0.0, elapsed_hours - time_since_arrival), elapsed_hours


if __name__ == '__main__':
    main()
