import torch
import torch.nn as nn

from .attention import SimpleAttention
from .feature_attention import FeatureAttention
from .temporal_attention import TemporalAttention


class EncoderBlock(nn.Module):
    """
    Transformer-style encoder block:
      - MH Self-Attention + Residual + LayerNorm
      - Optional Temporal Attention (time-step emphasis)
      - Optional Feature Attention (channel emphasis)
      - FFN + Residual + LayerNorm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        use_temporal_attention: bool = True,
        use_feature_attention: bool = True,
    ):
        super().__init__()

        self.attention = SimpleAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.temporal_attention = TemporalAttention(d_model) if use_temporal_attention else nn.Identity()
        self.feature_attention = FeatureAttention(d_model) if use_feature_attention else nn.Identity()

        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        attn_output = self.attention(x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        x = self.temporal_attention(x)
        x = self.feature_attention(x)

        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x


class Encoder(nn.Module):
    """
    Stack of encoder blocks.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int = 4,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        use_temporal_attention: bool = True,
        use_feature_attention: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    use_temporal_attention=use_temporal_attention,
                    use_feature_attention=use_feature_attention,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return x
