import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """
    Learn dynamic importance weights across time steps.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        weights = self.score(x)          # (B, T, 1)
        weights = torch.softmax(weights, dim=1)
        context = torch.sum(x * weights, dim=1, keepdim=True)  # (B,1,D)
        context = context.repeat(1, x.size(1), 1)
        return x + context