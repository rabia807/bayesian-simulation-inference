"""Run the complete simulation-based inference demonstration."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.simulation import make_training_data, sample_prior, simulate
from src.train import train_posterior
from src.inference import infer_posterior

N_SIMULATIONS = 12000
N_OBSERVATIONS = 80
EPOCHS = 35
SEED = 7

def main():
    results = Path("results")
    results.mkdir(exist_ok=True)

    print("Generating simulator training data...")
    observations, theta = make_training_data(
        N_SIMULATIONS, N_OBSERVATIONS, seed=SEED
    )
    print("Training neural posterior estimator...")
    model, metadata, history = train_posterior(
        observations, theta, epochs=EPOCHS, seed=SEED
    )

    # A fresh test case with known parameters.
    rng = np.random.default_rng(SEED + 1)
    true_theta = np.array([1.25, np.log(0.65)], dtype=np.float32)
    observed = simulate(true_theta, N_OBSERVATIONS, rng)
    post_mean, post_std = infer_posterior(model, metadata, observed)
    post_mean, post_std = post_mean.reshape(2), post_std.reshape(2)

    labels = ["mu", "log(sigma)"]
    truth = true_theta
    metrics = {
        "true_mu": float(truth[0]),
        "true_sigma": float(np.exp(truth[1])),
        "posterior_mean_mu": float(post_mean[0]),
        "posterior_mean_sigma": float(np.exp(post_mean[1])),
        "posterior_sd_mu": float(post_std[0]),
        "posterior_sd_log_sigma": float(post_std[1]),
        "device": metadata["device"],
    }
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    plt.figure(figsize=(7, 4))
    plt.plot(history["train"], label="train")
    plt.plot(history["validation"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Gaussian NLL (standardized parameters)")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results / "training_loss.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, ax in enumerate(axes):
        grid = np.linspace(post_mean[i] - 4 * post_std[i],
                           post_mean[i] + 4 * post_std[i], 300)
        density = np.exp(-0.5 * ((grid - post_mean[i]) / post_std[i]) ** 2)
        density /= (post_std[i] * np.sqrt(2 * np.pi))
        ax.plot(grid, density)
        ax.axvline(truth[i], linestyle="--", label="true value")
        ax.axvline(post_mean[i], linestyle=":", label="posterior mean")
        ax.set_title(labels[i])
        ax.set_xlabel("Parameter value")
        ax.set_ylabel("Approximate density")
        ax.legend()
    fig.suptitle("Neural approximate posterior")
    fig.tight_layout()
    fig.savefig(results / "posterior_estimate.png", dpi=160)
    plt.close(fig)
    print("Done. See the results/ directory.")

if __name__ == "__main__":
    main()
