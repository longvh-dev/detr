from typing import Callable
from copy import deepcopy

import torch.nn.functional as F
import torch.nn as nn


def _get_clones(module: Callable, N: int) -> nn.ModuleList:
    """Produce N identical layers."""
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation: str) -> Callable:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "glu":
        return F.glu
    else:
        raise ValueError(f"activation should be relu/gelu, not {activation}.")
