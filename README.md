# Bayesian Simulation-Based Inference with Neural Networks

A small, runnable project demonstrating **amortized simulation-based inference (SBI)**. It uses a simulator to generate datasets from a prior, then trains a neural network to approximate the posterior over the unknown parameters of a Gaussian data-generating process.

## Model

For observations \(x_1,\ldots,x_n\):

\[
x_i \sim \mathcal{N}(\mu,\sigma^2)
\]

The prior is \(\mu \sim U(-3,3)\) and \(\log \sigma \sim U(\log 0.2,\log 2)\). The network receives summary statistics (sample mean, log sample standard deviation, and standardized quantiles) and predicts a diagonal Gaussian approximation to the posterior over \((\mu,\log\sigma)\).

This is a compact educational SBI example, not a general-purpose posterior sampler. The diagonal Gaussian approximation may miss posterior skewness or dependence.

## Quick start

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\\Scripts\\activate

pip install -r requirements.txt
python main.py
```

The run trains the network and writes:
- `results/training_loss.png`
- `results/posterior_estimate.png`
- `results/metrics.json`

## Project layout

```text
bayesian-simulation-inference/
├── src/
│   ├── __init__.py
│   ├── simulation.py
│   ├── features.py
│   ├── model.py
│   ├── train.py
│   └── inference.py
├── results/                 # generated outputs
├── main.py
├── requirements.txt
└── .gitignore
```

## What to experiment with

1. Increase `N_SIMULATIONS` in `main.py`.
2. Change the prior ranges in `src/simulation.py`.
3. Increase the number of observations per simulated dataset.
4. Compare the neural approximation with a grid-based posterior for a single dataset.
5. Replace the Gaussian simulator with a more complex stochastic model.

## Reproducibility

The example uses fixed random seeds. CPU execution is supported; CUDA is used automatically when available.
