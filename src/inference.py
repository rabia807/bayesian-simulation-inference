"""Posterior prediction for a newly observed dataset."""
import numpy as np
import torch
from .features import summarize

@torch.no_grad()
def infer_posterior(model, metadata, observations):
    features = summarize(observations)
    features = (features - metadata["x_mean"]) / metadata["x_std"]
    device = next(model.parameters()).device
    x = torch.tensor(features, dtype=torch.float32, device=device)
    model.eval()
    mean_z, log_std_z = model(x)
    mean = mean_z.cpu().numpy() * metadata["y_std"] + metadata["y_mean"]
    std = (torch.exp(log_std_z).cpu().numpy() * metadata["y_std"])
    return mean, std
