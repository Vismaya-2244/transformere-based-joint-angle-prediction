from typing import Optional

import torch
import torch.nn as nn


class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block with masked self-attention + cross-attention."""

    def __init__(self, d_model: int, num_heads: int = 4, ffn_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Masked self-attention on decoder targets
        self_attn_out, _ = self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt,
            attn_mask=tgt_mask,
            need_weights=False,
        )
        tgt = self.norm1(tgt + self.dropout1(self_attn_out))

        # Cross-attention: decoder queries attend to encoder memory
        cross_attn_out, _ = self.cross_attn(
            query=tgt,
            key=memory,
            value=memory,
            need_weights=False,
        )
        tgt = self.norm2(tgt + self.dropout2(cross_attn_out))

        # FFN
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout3(ffn_out))
        return tgt


class TransformerDecoder(nn.Module):
    """Stacked transformer decoder for multi-step regression outputs."""

    def __init__(
        self,
        d_model: int,
        future_steps: int,
        output_dim: int = 1,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.output_dim = output_dim

        # Learned query tokens for inference when teacher input is not provided
        self.future_queries = nn.Parameter(torch.randn(1, future_steps, d_model) * 0.02)

        # Optional teacher-forcing projection from output space to model space
        self.target_proj = nn.Linear(output_dim, d_model)
        self.target_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.out_head = nn.Linear(d_model, output_dim)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # shape: (T, T), upper-triangular entries masked
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, memory: torch.Tensor, target_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = memory.size(0)

        if target_seq is not None:
            tgt = self.target_proj(target_seq)
            tgt = self.target_dropout(tgt)
        else:
            tgt = self.future_queries.expand(batch_size, -1, -1)

        tgt_mask = self._causal_mask(tgt.size(1), tgt.device)

        for layer in self.layers:
            tgt = layer(tgt=tgt, memory=memory, tgt_mask=tgt_mask)

        return self.out_head(tgt)