# imprint/protos.py

from __future__ import annotations
from typing import Optional, Tuple, List
import torch
import torch.nn as nn


class GRUStack(nn.Module):
    """
    Prototype: stacked GRUs with optional layer normalization.

    Responsibilities:
      - Maintain recurrent hidden state.
      - Implement a single-step API: given input drive at this tick and previous
        hidden state, produce new state and output activations per layer.
    """

    def __init__(
        self,
        hidden: int,
        layers: int = 1,
        layernorm: bool = False,
        input_size: Optional[int] = None,  # may be set at bind() if None
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.layernorm = layernorm
        self.input_size = input_size
        # Internal: GRU layers, norms, etc.

    def init_state(self, batch_size: int) -> torch.Tensor:
        """Return initial hidden state [layers, B, hidden]."""
        ...

    def step(
        self,
        drive: torch.Tensor,      # [B, D_in]
        state: torch.Tensor,      # [layers, B, hidden]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single recurrent step.

        Returns:
          new_state: [layers, B, hidden]
          output:    [B, hidden] (e.g., top layer output)
        """
        ...


class MLP(nn.Module):
    """
    Prototype: simple multilayer perceptron.

    Responsibilities:
      - Map from input_dim to output_dim through linear+nonlinearity stack.
      - input_dim / output_dim may be Auto and resolved at bind().
    """

    def __init__(
        self,
        widths: List[int],      # e.g., [in_dim, hidden, out_dim] or [hidden1, out_dim] if input inferred
        act: str = "relu",
    ) -> None:
        super().__init__()
        self.widths = widths
        self.act = act
        # Internal setup deferred until bind() if needed.


class Aggregator(nn.Module):
    """
    Prototype: combine multiple input subports into an aggregated representation.

    Responsibilities:
      - Accept dict-like input of subport tensors: {'deep': ..., 'wide': ..., ...}.
      - Combine them (concat, sum, attention, etc.), then map to out_dim.
    """

    def __init__(
        self,
        mode: str = "concat→linear",
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.out_dim = out_dim
        # Internal layers set up at bind().


class Elementwise(nn.Module):
    """
    Prototype: simple elementwise operations between two inputs.

    Responsibilities:
      - Implement operations like 'sub→abs', 'sub→square', etc.
    """

    def __init__(self, op: str) -> None:
        super().__init__()
        self.op = op

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Apply op elementwise to (a, b).
        """
        ...
