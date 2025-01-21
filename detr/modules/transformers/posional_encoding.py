"""
Posional Encoding Module for Transformer
"""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional encoding module for transformer.
    """

    def __init__(self, d_model: int, max_len: int) -> None:
        """Initialize the `PositionalEncoding` module.

        :param d_model: The model dimensionality.
        :param max_len: The maximum length of the input sequence.
        :param device: The device to use.
        """

        super().__init__()

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the `PositionalEncoding` module.

        :param x: The input tensor.
        :return: The tensor with positional encoding added.
        """
        return x + self.pe[: x.size(0), :]
