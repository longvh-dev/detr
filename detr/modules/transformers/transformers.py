"""
DETR Transformers Module

References the torch.nn.transformers module and provides additional:
- Positional Encoding is passed in Multi Head Attention
- Extra Linear layer at end of encoder is removed
- Decoder returns a stack of activations from all decoding layer
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from detr.modules.transformers.encoder import (
    TransformerEncoder,
    TransformerEncoderLayer,
)
from detr.modules.transformers.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)


class Transformer(nn.Module):
    """Transformer Module"""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
        return_intermediate_dec: bool = False,
    ) -> None:
        """Initialize the `Transformer` module.

        :param d_model: The model dimensionality.
        :param nhead: The number of heads in the multiheadattention models.
        :param num_encoder_layers: The number of sub-encoder-layers in the encoder.
        :param num_decoder_layers: The number of sub-decoder-layers in the decoder.
        :param dim_feedforward: The dimension of the feedforward network model.
        :param dropout: The dropout value.
        :param activation: The activation function to use.
        :param normalize_before: Whether to apply layer normalization before the sublayer.
        :param return_intermediate_dec: Whether to return the intermediate activations.
        """
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation, normalize_before
            ),
            num_encoder_layers,
            norm=encoder_norm,
        )

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation, normalize_before
            ),
            num_decoder_layers,
            norm=decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        query_embed: Optional[torch.Tensor] = None,
        pos_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the `Transformer` module.

        :param src: The input sequence to the encoder (required) (NxCxHxW).
        :param mask: The mask for the src sequence (optional) (NxHxW).
        :param query_embed: The positional encoding for the query (optional) (NxC).
        :param pos_embed: The positional encoding for the tgt sequence (optional) (NxHxW).
        :return: The output of the transformer (NHxWxC).
        """
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


if __name__ == "__main__":
    transformer = Transformer()
    print(transformer.d_model)
