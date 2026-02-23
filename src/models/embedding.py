import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.linear = nn.Linear(input_dim, d_model)

    def forward(self, x):
        """
        x: (batch, past_steps, input_dim)
        returns: (batch, past_steps, d_model)
        """
        return self.linear(x)