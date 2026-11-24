"""Neural network predicting a Gaussian approximation to the posterior."""
import torch
from torch import nn

class PosteriorNet(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=96):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        raw = self.head(self.backbone(x))
        mean = raw[..., :2]
        # Bound log standard deviations for stable training.
        log_std = raw[..., 2:].clamp(-4.0, 1.5)
        return mean, log_std

def gaussian_nll(pred_mean, pred_log_std, target):
    inv_var = torch.exp(-2.0 * pred_log_std)
    return (0.5 * (target - pred_mean).square() * inv_var
            + pred_log_std).sum(dim=-1).mean()
