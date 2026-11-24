# Full simulated dataset (12,000 simulations)

This folder contains the complete 12,000-row synthetic dataset in 12 CSV shards.

- Files: `simulations_01.csv` through `simulations_12.csv`
- Rows per file: 1,000 simulations (plus header)
- Total simulations: 12,000
- Observations per simulation: 80
- Columns: `simulation_id`, `mu`, `log_sigma`, `sigma`, and `x1` through `x80`
- Parameter ranges: `mu ~ Uniform(-3, 3)`; `log_sigma ~ Uniform(log(0.2), log(2.0))`
- Observations are generated as independent draws from `Normal(mu, sigma^2)`.

The CSVs are split into shards to make downloading and browsing manageable. They are a standalone generated dataset from the same model and prior ranges used by the project simulator. They are not byte-for-byte output from the NumPy seed-7 run performed by `main.py`; run `python main.py` to regenerate the training data with that exact code path.
