# Dataset

`simulated_dataset_sample.csv` contains 100 reproducible example simulations from the Gaussian simulator used in this project.

Each row represents one simulated dataset and includes:
- `simulation_id`: row identifier
- `mu`: sampled mean parameter
- `log_sigma`: sampled log standard deviation
- `sigma`: standard deviation on its original scale
- `x1` through `x80`: the 80 simulated observations

The full training dataset is generated in memory by `main.py` using 12,000 simulations. It is not committed because it can be regenerated from the simulator and would make the repository unnecessarily large.
