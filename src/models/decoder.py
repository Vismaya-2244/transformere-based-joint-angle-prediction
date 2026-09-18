import torch
import torch.nn as nn


class SimpleDecoder(nn.Module):
    """
    Decoder head for multi-step future regression.

    Strategy:
      - use mean-pooled encoder representation across all timesteps
      - map to full future sequence in one shot
    """

    def __init__(self, d_model: int, future_steps: int, output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.future_steps = future_steps
        self.output_dim = output_dim

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, future_steps * output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) encoder output
        Returns:
            (batch, future_steps, output_dim)
        """
        x_ctx = torch.mean(x, dim=1)  # (B, D)

        out = self.head(x_ctx)  # (B, future_steps * output_dim)
        out = out.view(x_ctx.size(0), self.future_steps, self.output_dim)
        return out