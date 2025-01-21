"""
Decoder module.
- Decoder returns a stack of activations from all decoding layers.
"""

from typing import Optional

import torch
import torch.nn as nn

from detr.modules.transformers.utils import _get_activation_fn, _get_clones


class TransformerDecoder(nn.Module):
    """Transformer Decoder Module."""

    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int = 6,
        norm: nn.Module = None,
        return_intermediate: bool = False,
    ) -> None:
        """Initialize the `TransformerDecoder` module.

        :param decoder_layer: An instance of the `TransformerDecoderLayer` module.
        :param num_layers: The number of sub-decoder-layers in the decoder.
        :param norm: The layer normalization module.
        :param return_intermediate: Whether to return the intermediate activations
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the `TransformerDecoder` module.

        :param tgt: The input sequence to the decoder (required).
        :param memory: The sequence from the last layer of the encoder (required).
        :param tgt_mask: The mask for the tgt sequence (optional).
        :param memory_mask: The mask for the memory sequence (optional).
        :param tgt_key_padding_mask: The mask for the tgt keys per batch (optional).
        :param memory_key_padding_mask: The mask for the memory keys per batch (optional).
        :param pos: The positional encoding for the tgt sequence (optional).
        :param query_pos: The positional encoding for the query (optional).
        :return: The output of the decoder.
        """
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer Module."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
    ) -> None:
        """Initialize the `TransformerDecoderLayer` module.

        :param d_model: The model dimensionality.
        :param nhead: The number of heads in the multiheadattention models.
        :param dim_feedforward: The dimension of the feedforward network model.
        :param dropout: The dropout value.
        :param activation: The activation function to use.
        :param normalize_before: Whether to apply layer normalization before the layer.
        """
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward Layer
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.normalize_before = normalize_before

    def with_pos_embed(
        self, tensor: torch.Tensor, pos: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ):
        x = tgt

        # Self-attention block
        if self.normalize_before:
            x2 = self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, query_pos=query_pos
            )
            x = x + x2
        else:
            x2 = self._sa_block(x, tgt_mask, tgt_key_padding_mask, query_pos=query_pos)
            x = self.norm1(x + x2)

        # Multi-head attention block
        if self.normalize_before:
            x2 = self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
            x = x + x2
        else:
            x2 = self._mha_block(
                x, memory, memory_mask, memory_key_padding_mask, pos, query_pos
            )
            x = self.norm2(x + x2)

        # Feedforward block
        if self.normalize_before:
            x2 = self._ff_block(self.norm3(x))
            x = x + x2
        else:
            x2 = self._ff_block(x)
            x = self.norm3(x + x2)

        return x

    def _sa_block(
        self,
        x: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Self-attention block."""
        q = k = self.with_pos_embed(x, query_pos)
        x2 = self.self_attn(
            q,
            k,
            value=x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        x = x + self.dropout1(x2)
        return x

    def _mha_block(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Multi-head attention block."""
        x2 = self.multihead_attn(
            query=self.with_pos_embed(x, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        x = x + self.dropout2(x2)
        return x

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Feedforward block."""
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout3(x2)
        return x
