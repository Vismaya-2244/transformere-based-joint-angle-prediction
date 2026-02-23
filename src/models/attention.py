import torch
import torch.nn as nn
import math

class SimpleAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        returns: (batch, seq_len, d_model)
        """

        Q = self.q(x)  # (batch, seq_len, d_model)
        K = self.k(x)
        V = self.v(x)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-1, -2))  # (batch, seq_len, seq_len)

        # Scale
        scores = scores / math.sqrt(x.size(-1))

        # Softmax
        weights = torch.softmax(scores, dim=-1)

        # Output
        output = torch.matmul(weights, V)  # (batch, seq_len, d_model)

        return output