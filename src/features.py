"""Low-dimensional summary statistics used by the neural estimator."""
import numpy as np

def summarize(data: np.ndarray) -> np.ndarray:
    """Map one dataset or a batch of datasets to stable summary features."""
    x = np.asarray(data)
    single = x.ndim == 1
    if single:
        x = x[None, :]
    mean = x.mean(axis=1)
    std = np.maximum(x.std(axis=1, ddof=1), 1e-6)
    q10, q50, q90 = np.quantile(x, [0.1, 0.5, 0.9], axis=1)
    # Quantiles are standardized to retain shape information beyond mean/std.
    features = np.column_stack([
        mean, np.log(std), (q10 - mean) / std,
        (q50 - mean) / std, (q90 - mean) / std
    ]).astype(np.float32)
    return features[0] if single else features
