"""
Encoder module
- extra linear layer at end of encoder is removed
"""

from typing import Optional

import torch
import torch.nn as nn

from detr.modules.transformers.utils import _get_activation_fn, _get_clones


class TransformerEncoder(nn.Module):
    """Transformer Encoder"""

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int = 6,
        norm: nn.Module = None,
    ) -> None:
        """Initialize the `TransformerEncoder` module.

        :param encoder_layer: An instance of the `TransformerEncoderLayer` module.
        :param num_layers: The number of sub-encoder-layers in the encoder.
        :param norm: The layer normalization module.
        """
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the `TransformerEncoder` module.

        :param src: The input sequence to the encoder (required).
        :param mask: The mask for the src sequence (optional).
        :param src_key_padding_mask: The mask for the src keys per batch (optional).
        :return: The output of the encoder.
        """
        output = src

        for layer in self.layers:
            output = layer(
                src=output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
    ) -> None:
        """Initialize the `TransformerEncoderLayer` module.

        :param d_model: The model dimensionality.
        :param nhead: The number of heads in the multiheadattention models.
        :param dim_feedforward: The dimension of the feedforward network model.
        :param dropout: The dropout value.
        :param activation: The activation function of intermediate layer, relu or gelu.
        :param normalize_before: Whether to apply layer normalization before the first block.
        """
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(
        self, tensor: torch.Tensor, pos: Optional[torch.Tensor]
    ) -> torch.Tensor:
        assert tensor.shape[1:] == pos.shape[1:]
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        q = k = self.with_pos_embed(src, pos)

        attn_output = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        feedforward_output = self.linear2(
            self.dropout(self.activation(self.linear1(src)))
        )
        src = src + self.dropout2(feedforward_output)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)

        attn_output = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(attn_output)

        src2 = self.norm2(src)
        feedforward_output = self.linear2(
            self.dropout(self.activation(self.linear1(src2)))
        )
        src = src + self.dropout2(feedforward_output)
        return src

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)
