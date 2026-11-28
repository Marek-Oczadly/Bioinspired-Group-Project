# Running the forest-fire simulation
## CaPyle interface

From the project root, run:

```sh
cd CA_tool
mkdir -p temp
python3 main.py
```

In CaPyle, choose **File → Open**, select `ca_descriptions/forest_fires.py`,
then click **Apply configuration & run CA**.

## Command line

From the project root, run:

```sh
cd CA_tool
python3 run_cli.py
```

To run a single simulation, use `python3 run_cli.py --runs 1`.

## Config
To change the scenario, edit [run_constants.json](CA_tool/ca_descriptions/forest_fires_meta/run_constants.json).

To test with water and forest, set guard to True, to test the default config, set guard to False.
