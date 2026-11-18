"""Run CaPyle simulations from the terminal."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import json
from pathlib import Path
import sys

# CaPyle import fix
ROOT = Path(__file__).resolve().parent
for folder in (ROOT, ROOT / 'capyle', ROOT / 'capyle/ca',
               ROOT / 'capyle/guicomponents'):
    sys.path.append(str(folder))

import numpy as np
from capyle.ca import CAConfig, Grid2D

# TODO: Import the forest-fire transition function.
from ca_descriptions.gol_2d import transition_func


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
            run_simulation(run, runs, generations, grid_size, seed)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, runs)) as pool:
            for _ in pool.map(run_simulation, range(runs), repeat(runs),
                              repeat(generations), repeat(grid_size), repeat(seed)):
                pass


def run_simulation(run, total, generations, grid_size, seed):
    # TODO: Load the forest-fire model configuration and states.
    config = CAConfig(str(ROOT / 'ca_descriptions/gol_2d.py'))
    config.states = (0, 1, 2)
    config.grid_dims = tuple(grid_size)
    config.num_generations = generations
    config.fill_in_defaults()

    run_seed = seed + run
    rng = np.random.default_rng(run_seed)
    # TODO: Initialise the forest-fire map and select the incinerator or power plant.
    config.initial_grid = rng.choice(config.states, size=config.grid_dims)
    simulation = Grid2D(config, transition_func)

    # step() advances the simulation without opening a progress window.
    print(f'Simulation {run + 1}/{total}: {generations} generations, seed {run_seed}')
    for _ in range(generations):
        simulation.step()

    # TODO: Report first town arrival when the forest-fire model is connected.
    print(f'Finished. Town arrival: {np.count_nonzero(simulation.grid == 1)}')

    # save in new file and overwrite if rerun
    filename = 'final_grid.npy' if total == 1 else 'final_grid_{}.npy'.format(run + 1)
    output = ROOT / filename
    np.save(output, simulation.grid)
    print('Saved final grid to {}'.format(output))


if __name__ == '__main__':
    main()
