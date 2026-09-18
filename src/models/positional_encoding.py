# src/positional_encoding.py
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # Handle odd d_model safely
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len {self.pe.size(1)} "
                f"in PositionalEncoding."
            )

        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)