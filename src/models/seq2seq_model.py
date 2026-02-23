import torch
import torch.nn as nn

from .embedding import InputEmbedding
from .positional_encoding import PositionalEncoding
from .encoder import EncoderBlock
from .decoder import SimpleDecoder

class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim, d_model, past_steps, future_steps, num_encoder_blocks=2):
        super().__init__()

        self.embedding = InputEmbedding(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=past_steps)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model) for _ in range(num_encoder_blocks)
        ])

        self.decoder = SimpleDecoder(d_model, future_steps, output_dim=1)

    def forward(self, x):
        """
        x: (batch, past_steps, input_dim)
        returns: (batch, future_steps, 1)
        """

        x = self.embedding(x)
        x = self.positional_encoding(x)

        for block in self.encoder_blocks:
            x = block(x)

        out = self.decoder(x)

        return out