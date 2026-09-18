# src/attention.py
import math
import torch
import torch.nn as nn


class SimpleAttention(nn.Module):
    """
    Multi-head self-attention (kept class name as SimpleAttention for compatibility).
    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, H, T, Dh)
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, T, Dh) -> (B, T, D)
        b, h, t, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * dh)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        """
        Args:
            x: (batch, seq_len, d_model)
            attn_mask: optional mask broadcastable to (batch, num_heads, seq_len, seq_len)
                       Values should be 0 for keep, -inf for masked.
        Returns:
            output: (batch, seq_len, d_model)
        """
        q = self._split_heads(self.q(x))
        k = self._split_heads(self.k(x))
        v = self._split_heads(self.v(x))

        # (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            scores = scores + attn_mask

        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # (B, H, T, Dh)
        context = torch.matmul(weights, v)
        context = self._merge_heads(context)

        output = self.out(context)
        output = self.proj_dropout(output)
        return output