import torch
import torch.nn as nn
from .attention import SimpleAttention

class EncoderBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.attention = SimpleAttention(d_model)
        self.norm1 = nn.LayerNorm(d_model)

        # Feed Forward Network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Attention block
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)

        # Feed-forward block
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)

        return x