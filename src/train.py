"""Training loop for the amortized posterior estimator."""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .features import summarize
from .model import PosteriorNet, gaussian_nll

def train_posterior(observations, theta, epochs=35, batch_size=256, seed=7):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = summarize(observations)
    # Standardize input features; persist these values for inference.
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0) + 1e-6
    x = (x - x_mean) / x_std

    # Standardize targets to improve optimization.
    y_mean = theta.mean(axis=0)
    y_std = theta.std(axis=0) + 1e-6
    y = (theta - y_mean) / y_std

    n = len(x)
    split = int(0.9 * n)
    permutation = np.random.permutation(n)
    train_idx, val_idx = permutation[:split], permutation[split:]

    train_ds = TensorDataset(torch.tensor(x[train_idx]), torch.tensor(y[train_idx]))
    val_x = torch.tensor(x[val_idx])
    val_y = torch.tensor(y[val_idx])
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PosteriorNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    history = {"train": [], "validation": []}

    for _ in range(epochs):
        model.train()
        running = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            mean, log_std = model(xb)
            loss = gaussian_nll(mean, log_std, yb)
            loss.backward()
            optimizer.step()
            running.append(loss.item())

        model.eval()
        with torch.no_grad():
            mean, log_std = model(val_x.to(device))
            val_loss = gaussian_nll(mean, log_std, val_y.to(device)).item()
        history["train"].append(float(np.mean(running)))
        history["validation"].append(float(val_loss))

    metadata = {
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_mean, "y_std": y_std,
        "device": str(device)
    }
    return model, metadata, history
