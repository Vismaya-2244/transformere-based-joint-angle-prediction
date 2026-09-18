# src/embedding.py
import math
import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    Projects raw sensor features into model dimension.
    """

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, seq_len, d_model)
        """
        x = self.proj(x) * self.scale
        x = self.norm(x)
        x = self.dropout(x)
        return x