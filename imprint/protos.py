# imprint/protos.py

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import Auto  # type: ignore  # Local import to avoid circular typing


class GRUStack(nn.Module):
    """
    Prototype: stacked GRUs with optional layer normalization.

    Provides a lightweight single-step API used by the scheduler.
    """

    def __init__(
        self,
        hidden: int,
        layers: int = 1,
        layernorm: bool = False,
        input_size: Optional[int] = None,
        reset_every: Optional[int] = None,
        out_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        self.hidden = hidden
        self.layers = layers
        self.layernorm = layernorm
        self.input_size = input_size
        if reset_every is not None and reset_every < 1:
            raise ValueError("reset_every must be >= 1 when provided")
        self.reset_every = int(reset_every) if reset_every is not None else None
        self._steps_since_reset = 0

        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList() if layernorm else None

        # Optional output projection (exposed on module 'out' port only; recurrent state remains 'hidden')
        self.out_dim: Optional[int] = int(out_dim) if out_dim is not None else None
        self.out_proj: Optional[nn.Linear] = None
        self.out_norm: Optional[nn.LayerNorm] = None
        if input_size is not None:
            self._build_cells(input_size)

    def _build_cells(self, input_size: int) -> None:
        prev_dim = input_size
        self.cells = nn.ModuleList()
        for _ in range(self.layers):
            self.cells.append(nn.GRUCell(prev_dim, self.hidden))
            prev_dim = self.hidden
        if self.layernorm:
            self.norms = nn.ModuleList([nn.LayerNorm(self.hidden) for _ in range(self.layers)])
        # Build output projection lazily later in bind()

    def bind(self, input_dim: int) -> None:
        if self.input_size is not None and self.input_size != input_dim:
            raise ValueError(f"GRUStack expected input_dim {self.input_size}, got {input_dim}")
        if self.input_size is None:
            self.input_size = input_dim
            self._build_cells(input_dim)
        # Materialize output projection after input is known
        if self.out_dim is not None and self.out_proj is None:
            self.out_proj = nn.Linear(self.hidden, self.out_dim)
            if self.layernorm:
                self.out_norm = nn.LayerNorm(self.out_dim)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        if self.input_size is None:
            raise RuntimeError("GRUStack.bind() must be called before init_state.")
        shape = (self.layers, batch_size, self.hidden)
        self._steps_since_reset = 0
        return torch.zeros(shape, device=device)

    def step(
        self,
        drive: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Single recurrent step.

        Returns:
            new_state: [layers, B, hidden]
            outputs: list of per-layer outputs, each [B, hidden].
        """
        if not self.cells:
            raise RuntimeError("GRUStack.bind() must be called before step().")

        batch_size = drive.shape[0]
        if state.shape[0] != self.layers or state.shape[1] != batch_size:
            raise ValueError("State shape mismatch for GRUStack.")

        layer_outputs: List[torch.Tensor] = []
        next_states = []
        layer_input = drive
        for idx, cell in enumerate(self.cells):
            layer_state = state[idx]
            updated = cell(layer_input, layer_state)
            if self.norms is not None:
                updated = self.norms[idx](updated)
            layer_outputs.append(updated)
            next_states.append(updated)
            layer_input = updated

        stacked = torch.stack(next_states, dim=0)
        if self.reset_every is not None:
            self._steps_since_reset += 1
            if self._steps_since_reset >= self.reset_every:
                stacked = torch.zeros_like(stacked)
                self._steps_since_reset = 0
        return stacked, layer_outputs

    # Optional hook for Module to expose a transformed output port without
    # altering layered hidden states.
    def expose_output(self, last_hidden: torch.Tensor) -> torch.Tensor:
        x = last_hidden
        if self.out_proj is not None:
            x = self.out_proj(x)
            if self.out_norm is not None:
                x = self.out_norm(x)
        return x


class MLP(nn.Module):
    """
    Prototype: simple multilayer perceptron supporting Auto dims.
    """

    def __init__(
        self,
        widths: List[int],
        act: str = "relu",
    ) -> None:
        super().__init__()
        if not widths:
            raise ValueError("widths must contain at least one layer size.")
        self.widths = list(widths)
        self.act = act
        self.layers = nn.ModuleList()
        self._bound = False
        self.output_dim: Optional[int] = None

    def _resolve_widths(self, input_dim: int, output_dim: Optional[int]) -> List[int]:
        dims = []
        for idx, width in enumerate(self.widths):
            if width is Auto:
                if idx == len(self.widths) - 1:
                    if output_dim is None:
                        raise ValueError("Output dimension must be specified for Auto width.")
                    dims.append(int(output_dim))
                else:
                    raise ValueError("Auto widths are only supported for the final layer.")
            else:
                dims.append(int(width))
        resolved = [input_dim] + dims
        return resolved

    def bind(self, input_dim: int, output_dim: Optional[int]) -> None:
        if self._bound:
            expected_out = self.output_dim
            if expected_out is not None and output_dim is not None and expected_out != output_dim:
                raise ValueError("Cannot re-bind MLP with a different output dimension.")
            return
        dims = self._resolve_widths(input_dim, output_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.output_dim = dims[-1]
        self._bound = True

    def _activation(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.act == "relu":
            return F.relu(tensor)
        if self.act == "gelu":
            return F.gelu(tensor)
        if self.act == "tanh":
            return torch.tanh(tensor)
        if self.act == "identity":
            return tensor
        raise ValueError(f"Unsupported activation {self.act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.layers:
            raise RuntimeError("MLP.bind() must be called before forward().")
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = self._activation(x)
        return x


class Aggregator(nn.Module):
    """
    Prototype: combine multiple input subports into an aggregated representation.
    """

    def __init__(
        self,
        mode: str = "concat→linear",
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.out_dim = out_dim
        self.linear: Optional[nn.Linear] = None
        self._input_order: List[str] = []
        self._input_dims: Dict[str, int] = {}

    def bind(self, input_dims: Dict[str, int]) -> None:
        if not input_dims:
            raise ValueError("Aggregator requires at least one input.")
        self._input_dims = dict(input_dims)
        self._input_order = list(input_dims.keys())
        mode = self.mode.split("→")[0]
        if mode == "concat":
            total = sum(input_dims.values())
        elif mode == "sum":
            dims = list(input_dims.values())
            if len(set(dims)) != 1:
                raise ValueError("sum mode requires all input dims to match.")
            total = dims[0]
        else:
            raise ValueError(f"Unsupported aggregator mode {self.mode}")

        self.linear = nn.Linear(total, self.out_dim)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.linear is None:
            raise RuntimeError("Aggregator.bind() must be called before forward().")
        tensors: List[torch.Tensor] = []
        mode = self.mode.split("→")[0]
        for key in self._input_order:
            if key not in inputs:
                raise KeyError(f"Missing subport {key} for Aggregator forward.")
            tensors.append(inputs[key])

        if mode == "concat":
            combined = torch.cat(tensors, dim=-1)
        else:  # sum
            combined = torch.stack(tensors, dim=0).sum(dim=0)
        return self.linear(combined)


class FiLMConditioner(nn.Module):
    """
    Prototype: FiLM modulation driven by a conditioning signal.
    """

    def __init__(self, widths: List[int], act: str = "relu") -> None:
        super().__init__()
        if not widths:
            raise ValueError("widths must contain at least one layer size.")
        self.widths = list(widths)
        self.act = act
        self.mode = "film"
        self.layers = nn.ModuleList()
        self.signal_dim: Optional[int] = None
        self.cond_dim: Optional[int] = None
        self._bound = False

    def _resolve_widths(self, cond_dim: int, signal_dim: int) -> List[int]:
        dims = [cond_dim]
        for idx, width in enumerate(self.widths):
            if width is Auto:
                if idx != len(self.widths) - 1:
                    raise ValueError("Auto widths supported only on the final FiLM layer.")
                dims.append(2 * signal_dim)
            else:
                dims.append(int(width))
        if dims[-1] != 2 * signal_dim:
            raise ValueError("FiLMConditioner final width must equal 2 * signal dimension.")
        return dims

    def bind(self, input_dims: Dict[str, int]) -> None:
        if "signal" not in input_dims or "cond" not in input_dims:
            raise ValueError("FiLMConditioner requires 'signal' and 'cond' inputs.")
        signal_dim = int(input_dims["signal"])
        cond_dim = int(input_dims["cond"])
        if signal_dim <= 0 or cond_dim <= 0:
            raise ValueError("FiLMConditioner input dims must be positive.")
        if self._bound:
            if signal_dim != self.signal_dim or cond_dim != self.cond_dim:
                raise ValueError("Cannot re-bind FiLMConditioner with different dimensions.")
            return
        dims = self._resolve_widths(cond_dim, signal_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.signal_dim = signal_dim
        self.cond_dim = cond_dim
        self._bound = True

    def _activation(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.act == "relu":
            return F.relu(tensor)
        if self.act == "gelu":
            return F.gelu(tensor)
        if self.act == "tanh":
            return torch.tanh(tensor)
        if self.act == "identity":
            return tensor
        raise ValueError(f"Unsupported activation {self.act}")

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.layers or self.signal_dim is None:
            raise RuntimeError("FiLMConditioner.bind() must be called before forward().")
        if "signal" not in inputs or "cond" not in inputs:
            raise KeyError("FiLMConditioner forward requires 'signal' and 'cond' tensors.")
        signal = inputs["signal"]
        cond = inputs["cond"]
        x = cond
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = self._activation(x)
        gamma_raw, beta = torch.split(x, self.signal_dim, dim=-1)
        scale = torch.sigmoid(gamma_raw) * 2.0  # confine scale to (0, 2) while centering at 1
        return signal * scale + beta

    def infer_output_dim(self, input_dims: Dict[str, int]) -> Optional[int]:
        signal_dim = input_dims.get("signal")
        if signal_dim is None:
            signal_dim = input_dims.get("in.signal")
        if signal_dim is not None:
            return int(signal_dim)
        if self.signal_dim is not None:
            return self.signal_dim
        return None

class Elementwise(nn.Module):
    """
    Prototype: simple elementwise operations between two inputs.
    """

    def __init__(self, op: str) -> None:
        super().__init__()
        self.op = op

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        op = self.op
        if op == "sub→abs":
            return torch.abs(a - b)
        if op == "sub→square":
            diff = a - b
            return diff * diff
        if op == "add":
            return a + b
        if op == "sub":
            return a - b
        if op == "mul":
            return a * b
        raise ValueError(f"Unsupported Elementwise op {op}")

    def infer_output_dim(self, input_dims: Dict[str, int]) -> Optional[int]:
        for dim in input_dims.values():
            return dim
        return None
