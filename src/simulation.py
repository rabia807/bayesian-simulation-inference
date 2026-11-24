"""Prior sampling and stochastic data simulation."""
import numpy as np

MU_LOW, MU_HIGH = -3.0, 3.0
LOG_SIGMA_LOW, LOG_SIGMA_HIGH = np.log(0.2), np.log(2.0)

def sample_prior(rng: np.random.Generator, n: int):
    """Draw n parameter pairs; return columns [mu, log_sigma]."""
    mu = rng.uniform(MU_LOW, MU_HIGH, size=n)
    log_sigma = rng.uniform(LOG_SIGMA_LOW, LOG_SIGMA_HIGH, size=n)
    return np.column_stack((mu, log_sigma)).astype(np.float32)

def simulate(theta: np.ndarray, n_observations: int, rng: np.random.Generator):
    """Simulate one dataset for theta=[mu, log_sigma]."""
    mu, log_sigma = map(float, theta)
    return rng.normal(mu, np.exp(log_sigma), size=n_observations)

def make_training_data(n_simulations: int, n_observations: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    theta = sample_prior(rng, n_simulations)
    observations = np.stack([
        simulate(t, n_observations, rng) for t in theta
    ])
    return observations, theta
