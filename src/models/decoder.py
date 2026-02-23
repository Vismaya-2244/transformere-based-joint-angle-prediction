import torch
import torch.nn as nn

class SimpleDecoder(nn.Module):
    def __init__(self, d_model, future_steps, output_dim=1):
        super().__init__()

        self.future_steps = future_steps
        self.output_dim = output_dim

        self.fc = nn.Linear(d_model, future_steps * output_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        returns: (batch, future_steps, output_dim)
        """

        # Use last timestep
        x = x[:, -1, :]  # (batch, d_model)

        # Predict full future sequence
        out = self.fc(x)  # (batch, future_steps * output_dim)

        out = out.view(x.size(0), self.future_steps, self.output_dim)

        return out