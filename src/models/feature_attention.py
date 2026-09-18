import torch
import torch.nn as nn


class FeatureAttention(nn.Module):
    """
    Learn importance weights across feature channels.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        weights = torch.sigmoid(self.fc(x))
        return x + x * weights