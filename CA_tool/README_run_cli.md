# Forest-fire command-line runner

`run_cli.py` runs the CaPyle forest-fire model from the terminal. It prints a result for each run in **simulated hours**.

## Run it

Use the same Python environment as CaPyle. The runner needs NumPy, Matplotlib and Tkinter through CaPyle's imports. From the `CA_tool` folder, run:

```sh
python3 run_cli.py
```

For a quick check or a parallel batch:

```sh
python3 run_cli.py --runs 1 --generations 10 --workers 1
python3 run_cli.py --runs 12 --workers 4
```

## Arguments

| Argument | What it does |
| --- | --- |
| `--config PATH` | Load a different batch settings JSON file. The default is [simulation_config.json](simulation_config.json), beside the runner. Relative paths you pass here start from your current directory. |
| `--runs N` | Run the scenario `N` times. |
| `--generations N` | Run at most `N` simulation steps per run. The length of a step depends on the scenario, so the same limit can cover different amounts of simulated time. |
| `--workers N` | Run up to `N` simulations at once using separate processes. Use `1` to run them sequentially. The runner caps workers at the number of runs. |
| `-h`, `--help` | Show the available arguments and exit. |

The numeric arguments must be positive integers and override the JSON values for that command. The batch JSON also sets `grid_size` (must match the map) and `seed` (a non-negative integer). Run 1 uses the base seed, run 2 uses the base seed plus one, and so on.

To change the scenario itself, such as ignition sources, wind or the water budget, edit [run_constants.json](ca_descriptions/forest_fires_meta/run_constants.json).

## Results

A run stops when the first town cell ignites, when no cells are burning, or when the generation limit is reached. If the fire reaches town, the result is the first ignition time, including arrivals partway through a step. Otherwise, the runner reports how many simulated hours elapsed before the run ended.
