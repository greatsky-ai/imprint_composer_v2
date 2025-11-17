from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import init as nn_init

from .registry import register_backbone


@register_backbone("slicer")
class SlicerBackbone(nn.Module):
    """Two-stage RNN backbone: slicer stage (chunked) then context stage.

    Inputs: [B, T, in_dim]
    Outputs: [B, H_ctx] (sequence-level embedding suitable for classification/regression)

    The slicer stage splits the time dimension into `num_chunks` segments (equal split
    by default) and encodes each chunk with a stacked GRU, keeping only the final
    hidden state per chunk to form [B, N, H_slicer]. Optionally a linear projection
    maps chunk embeddings to `proj_dim`.

    The context stage is a stacked GRU over the N chunk embeddings; the final hidden
    state of the last GRU layer is returned as [B, H_ctx].
    """

    def __init__(
        self,
        in_dim: int,
        *,
        # Slicer stage
        num_chunks: int = 8,
        slicer_hidden_dim: int = 256,
        slicer_layers: int = 1,
        chunk_overlap_fraction: float = 0.0,
        proj_dim: Optional[int] = None,
        # Context stage
        context_hidden_dim: int = 256,
        context_layers: int = 1,
        # Common
        dropout: float = 0.1,
        norm: Optional[str] = None,
        simple_gru_mode: bool = False,
        # Initialization knobs
        weight_init: "str | None" = None,
        recurrent_weight_init: "str | None" = None,
        bias_init: "str | None" = None,
        recurrent_weight_scale: float = 1.0,
    ) -> None:
        super().__init__()

        if num_chunks < 1:
            raise ValueError("num_chunks must be >= 1")
        self.num_chunks = int(num_chunks)
        self.chunk_overlap_fraction = float(max(0.0, min(1.0, chunk_overlap_fraction)))

        # ----- Slicer RNN (stack of single-layer GRUs) -----
        slicer_cells = []
        for layer_idx in range(slicer_layers):
            input_size = in_dim if layer_idx == 0 else slicer_hidden_dim
            slicer_cells.append(
                nn.GRU(
                    input_size=input_size,
                    hidden_size=slicer_hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    dropout=0.0,
                )
            )
        self.slicer_layers = nn.ModuleList(slicer_cells)
        # Per-layer post norms for slicer stack
        self.slicer_norms = nn.ModuleList()
        norm_choice = (norm or "none").lower()
        for _ in range(slicer_layers):
            if norm_choice in ("layernorm", "layer", "ln"):
                self.slicer_norms.append(nn.LayerNorm(slicer_hidden_dim))
            elif norm_choice in ("rmsnorm", "rms"):
                rms_cls = getattr(nn, "RMSNorm", None)
                if rms_cls is None:
                    raise ImportError("Requested norm='rmsnorm' but torch.nn.RMSNorm is not available in this PyTorch version.")
                self.slicer_norms.append(rms_cls(slicer_hidden_dim))  # type: ignore[call-arg]
            elif norm_choice in ("batchnorm", "batch", "bn"):
                self.slicer_norms.append(nn.BatchNorm1d(slicer_hidden_dim))
            else:
                self.slicer_norms.append(nn.Identity())
        self.proj = nn.Linear(slicer_hidden_dim, int(proj_dim)) if (proj_dim is not None) else None

        # ----- Context RNN (stack of single-layer GRUs) -----
        ctx_in_dim = int(proj_dim) if (proj_dim is not None) else slicer_hidden_dim
        context_cells = []
        for layer_idx in range(context_layers):
            input_size = ctx_in_dim if layer_idx == 0 else context_hidden_dim
            context_cells.append(
                nn.GRU(
                    input_size=input_size,
                    hidden_size=context_hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    dropout=0.0,
                )
            )
        self.context_layers = nn.ModuleList(context_cells)
        # Per-layer post norms for context stack
        self.context_norms = nn.ModuleList()
        for _ in range(context_layers):
            if norm_choice in ("layernorm", "layer", "ln"):
                self.context_norms.append(nn.LayerNorm(context_hidden_dim))
            elif norm_choice in ("rmsnorm", "rms"):
                rms_cls = getattr(nn, "RMSNorm", None)
                if rms_cls is None:
                    raise ImportError("Requested norm='rmsnorm' but torch.nn.RMSNorm is not available in this PyTorch version.")
                self.context_norms.append(rms_cls(context_hidden_dim))  # type: ignore[call-arg]
            elif norm_choice in ("batchnorm", "batch", "bn"):
                self.context_norms.append(nn.BatchNorm1d(context_hidden_dim))
            else:
                self.context_norms.append(nn.Identity())

        self.drop = nn.Dropout(dropout)
        self.out_dim = int(context_hidden_dim)

        # ---- Optional simplification: disable the reset gate persistently ----
        self.simple_gru_mode = bool(simple_gru_mode)
        if self.simple_gru_mode:
            self._simple_gru_reset_bias_value = 10.0  # sigmoid(10) ~= 0.99995
            self._simple_gru_weight_constraints = []  # list[(param, mask)]
            self._simple_gru_bias_constraints = []  # list[(param, mask, add_vec)]
            self._register_simple_gru_constraints()
            self._enforce_simple_gru_constraints()

        # Parameter initialization
        self._initialize_parameters(
            weight_init=weight_init,
            recurrent_weight_init=recurrent_weight_init,
            bias_init=bias_init,
            recurrent_weight_scale=float(recurrent_weight_scale),
        )

    def _apply_norm(self, x: torch.Tensor, norm_layer: nn.Module) -> torch.Tensor:
        # x: [B, T, H]
        if isinstance(norm_layer, nn.BatchNorm1d):
            B, T, H = x.shape
            x2 = x.contiguous().view(B * T, H)
            x2 = norm_layer(x2)
            return x2.view(B, T, H)
        return norm_layer(x)

    def _chunk_sequence(self, x_emb: torch.Tensor) -> torch.Tensor:
        """Split [B, T, E] into N chunks of length L (ceil), pad last as needed.

        Returns [B*N, L, E]. Supports simple overlapping via `chunk_overlap_fraction`.
        """
        B, T, E = x_emb.shape
        N = self.num_chunks
        L = (T + N - 1) // N  # ceil(T/N)
        if L <= 0:
            raise ValueError("Invalid computed chunk length")

        if self.chunk_overlap_fraction <= 1e-8:
            T_pad = N * L
            if T_pad != T:
                pad = x_emb.new_zeros((B, T_pad - T, E))
                x_emb = torch.cat([x_emb, pad], dim=1)
            return x_emb.view(B, N, L, E).reshape(B * N, L, E)

        # Overlapping windows
        stride = max(1, int(round(L * (1.0 - self.chunk_overlap_fraction))))
        chunks = []
        for i in range(N):
            start = i * stride
            if start > max(0, T - 1):
                start = max(0, T - L)
            end = start + L
            if end <= T:
                sub = x_emb[:, start:end, :]
            else:
                need = end - T
                pad = x_emb.new_zeros((B, need, E))
                sub = torch.cat([x_emb[:, start:T, :], pad], dim=1)
            chunks.append(sub)
        return torch.stack(chunks, dim=1).reshape(B * N, L, E)

    def _run_stacked_gru(self, cells: nn.ModuleList, norms: nn.ModuleList, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a stack of 1-layer GRUs over a sequence.

        Returns: (last_layer_output_seq, last_hidden_of_last_layer)
        Shapes: out_seq [B, T, H_last], h_last [B, H_last]
        """
        h = x
        last_hidden_from_norm: Optional[torch.Tensor] = None
        for idx, (cell, norm_layer) in enumerate(zip(cells, norms)):
            out, _h_n = cell(h)
            out_norm = self._apply_norm(out, norm_layer)
            # Capture the last hidden from the normalized sequence (not after dropout)
            if idx == len(cells) - 1:
                last_hidden_from_norm = out_norm[:, -1, :]
            # Apply dropout to the sequence output for feeding into next layer
            h = self.drop(out_norm)
        if last_hidden_from_norm is None:
            raise RuntimeError("Expected at least one GRU layer")
        return h, last_hidden_from_norm

    # ---- Simple GRU mode implementation ----
    def _register_simple_gru_constraints(self) -> None:
        # Apply constraints to all GRU cells in both slicer and context stacks.
        stacks = list(self.slicer_layers) + list(self.context_layers)
        self._simple_gru_weight_constraints = []
        self._simple_gru_bias_constraints = []

        for cell in stacks:
            hidden_size = cell.hidden_size
            for name, param in cell.named_parameters():
                if name.startswith("weight_"):
                    mask = torch.ones_like(param.data)
                    mask[:hidden_size, ...] = 0  # zero out reset gate rows
                    self._simple_gru_weight_constraints.append((param, mask))

                    def _weight_grad_mask_hook(grad, m=mask):
                        return grad * m.to(grad.device)

                    param.register_hook(_weight_grad_mask_hook)
                elif name.startswith("bias_"):
                    mask = torch.ones_like(param.data)
                    mask[:hidden_size] = 0  # no grads for reset gate bias

                    add_vec = torch.zeros_like(param.data)
                    if name.startswith("bias_ih_"):
                        add_vec[:hidden_size] = self._simple_gru_reset_bias_value
                    else:  # bias_hh_*
                        add_vec[:hidden_size] = 0.0

                    self._simple_gru_bias_constraints.append((param, mask, add_vec))

                    def _bias_grad_mask_hook(grad, m=mask):
                        return grad * m.to(grad.device)

                    param.register_hook(_bias_grad_mask_hook)

        # Ensure constraints are re-applied before every forward
        def _pre_hook(_module, _inputs):
            self._enforce_simple_gru_constraints()

        self._simple_gru_pre_hook_handle = self.register_forward_pre_hook(_pre_hook)

    # ---- Initialization helpers ----
    def _init_matrix(self, tensor: torch.Tensor, method: str | None) -> None:
        if tensor is None or method is None or method == "default":
            return
        name = str(method).lower()
        if name in ("xavier_uniform", "xavier-uniform", "xavier"):
            nn_init.xavier_uniform_(tensor)
        elif name in ("xavier_normal", "xavier-normal"):
            nn_init.xavier_normal_(tensor)
        elif name in ("kaiming_uniform", "kaiming-uniform", "kaiming"):
            nn_init.kaiming_uniform_(tensor, nonlinearity="relu")
        elif name in ("kaiming_normal", "kaiming-normal"):
            nn_init.kaiming_normal_(tensor, nonlinearity="relu")
        elif name in ("orthogonal", "orth"):
            nn_init.orthogonal_(tensor)
        elif name == "normal":
            nn_init.normal_(tensor, mean=0.0, std=0.02)
        elif name == "uniform":
            nn_init.uniform_(tensor, a=-0.05, b=0.05)
        else:
            return

    def _init_bias(self, tensor: torch.Tensor, method: str | None) -> None:
        if tensor is None:
            return
        name = (method or "zeros").lower()
        if name in ("zeros", "zero"):
            nn_init.zeros_(tensor)
        elif name in ("ones", "one"):
            nn_init.ones_(tensor)
        elif name == "normal":
            nn_init.normal_(tensor, mean=0.0, std=0.02)
        elif name == "uniform":
            nn_init.uniform_(tensor, a=-0.05, b=0.05)
        else:
            nn_init.zeros_(tensor)

    def _initialize_parameters(
        self,
        *,
        weight_init: str | None,
        recurrent_weight_init: str | None,
        bias_init: str | None,
        recurrent_weight_scale: float,
    ) -> None:
        w_init = weight_init or "xavier_uniform"
        hh_init = recurrent_weight_init or w_init
        b_init = bias_init or "zeros"
        scale = float(recurrent_weight_scale)

        # Slicer stack
        for cell in self.slicer_layers:
            for name, param in cell.named_parameters():
                if name.startswith("weight_ih_"):
                    self._init_matrix(param.data, w_init)
                elif name.startswith("weight_hh_"):
                    self._init_matrix(param.data, hh_init)
                    if scale != 1.0:
                        param.data.mul_(scale)
                elif name.startswith("bias_"):
                    self._init_bias(param.data, b_init)

        # Projection layer
        if self.proj is not None:
            self._init_matrix(self.proj.weight.data, w_init)
            if self.proj.bias is not None:
                self._init_bias(self.proj.bias.data, b_init)

        # Context stack
        for cell in self.context_layers:
            for name, param in cell.named_parameters():
                if name.startswith("weight_ih_"):
                    self._init_matrix(param.data, w_init)
                elif name.startswith("weight_hh_"):
                    self._init_matrix(param.data, hh_init)
                    if scale != 1.0:
                        param.data.mul_(scale)
                elif name.startswith("bias_"):
                    self._init_bias(param.data, b_init)

    @torch.no_grad()
    def _enforce_simple_gru_constraints(self) -> None:
        if not self.simple_gru_mode:
            return
        for param, mask in self._simple_gru_weight_constraints:
            param.data.mul_(mask.to(param.data.device))
        for param, mask, add_vec in self._simple_gru_bias_constraints:
            device = param.data.device
            param.data.mul_(mask.to(device))
            param.data.add_(add_vec.to(device))

    # ---- Optional parameter counting overrides ----
    @torch.no_grad()
    def count_parameters_total(self) -> int:
        total = 0
        if not self.simple_gru_mode:
            return sum(p.numel() for p in self.parameters())
        for param, mask in self._simple_gru_weight_constraints:
            total += int((mask != 0).sum().item())
        for param, mask, _add in self._simple_gru_bias_constraints:
            total += int((mask != 0).sum().item())
        constrained_params = {id(p) for p, _ in self._simple_gru_weight_constraints}
        constrained_params.update(id(p) for p, _m, _a in self._simple_gru_bias_constraints)
        for p in self.parameters():
            if id(p) not in constrained_params:
                total += p.numel()
        return int(total)

    @torch.no_grad()
    def count_parameters_trainable(self) -> int:
        total = 0
        if not self.simple_gru_mode:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        for param, mask in self._simple_gru_weight_constraints:
            if param.requires_grad:
                total += int((mask != 0).sum().item())
        for param, mask, _add in self._simple_gru_bias_constraints:
            if param.requires_grad:
                total += int((mask != 0).sum().item())
        constrained_params = {id(p) for p, _ in self._simple_gru_weight_constraints}
        constrained_params.update(id(p) for p, _m, _a in self._simple_gru_bias_constraints)
        for p in self.parameters():
            if id(p) not in constrained_params and p.requires_grad:
                total += p.numel()
        return int(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Slicer stage: chunk and encode
        x_chunks = self._chunk_sequence(x)  # [B*N, L, E]
        _out_seq, h_last = self._run_stacked_gru(self.slicer_layers, self.slicer_norms, x_chunks)  # h_last: [B*N, Hs]
        z = h_last  # chunk embedding per chunk
        B = x.shape[0]
        N = self.num_chunks
        z = z.view(B, N, -1)
        if self.proj is not None:
            z = self.proj(z)

        # Context stage: run over chunk embeddings sequence and take final hidden
        if len(self.context_layers) == 0:
            raise RuntimeError("context_layers must be >= 1")
        _out_ctx, h_ctx = self._run_stacked_gru(self.context_layers, self.context_norms, z)
        return self.drop(h_ctx)


__all__ = [
    "SlicerBackbone",
]


