import torch
import torch.nn as nn

from .embedding import InputEmbedding
from .positional_encoding import PositionalEncoding
from .encoder import Encoder
from .transformer_decoder import TransformerDecoder


class Seq2SeqModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        past_steps: int,
        future_steps: int,
        output_dim: int = 1,
        num_encoder_blocks: int = 2,
        num_decoder_layers: int = 2,
        num_heads: int = 4, 
        ffn_dim: int = 128,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        use_temporal_attention: bool = True,
        use_feature_attention: bool = True,
    ):
        super().__init__()

        self.use_positional_encoding = use_positional_encoding

        self.embedding = InputEmbedding(
            input_dim=input_dim,
            d_model=d_model,
            dropout=dropout,
        )

        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=past_steps,
        )

        self.encoder = Encoder(
            num_layers=num_encoder_blocks,
            d_model=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_temporal_attention=use_temporal_attention,
            use_feature_attention=use_feature_attention,
        )

        self.decoder = TransformerDecoder(
            d_model=d_model,
            future_steps=future_steps,
            output_dim=output_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, target_seq: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding(x)

        if self.use_positional_encoding:
            x = self.positional_encoding(x)

        memory = self.encoder(x)
        out = self.decoder(memory, target_seq=target_seq)
        return out
