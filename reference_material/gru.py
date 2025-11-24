from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import init as nn_init

from .registry import register_backbone
from .locked_dropout import LockedDropout


@register_backbone("gru")
class GRUBackbone(nn.Module):
    """Feature-only GRU backbone returning sequence features [B, T, H].

    Built as a stack of single-layer GRU cells with optional normalization
    after each layer, similar to the MinGRU backbone's flexible depth design.

    Args:
        in_dim: Input feature dimension per time step.
        hidden_dim: GRU hidden size.
        n_layers: Number of GRU layers.
        dropout: Dropout probability applied to outputs.
        norm: Optional normalization applied after each layer. One of
            {"layernorm"|"layer"|"ln", "rmsnorm"|"rms", "batchnorm"|"batch"|"bn"} or None.
        simple_gru_mode: If True, permanently disables the reset gate across all
            layers and directions: all weights feeding the reset gate are
            zeroed, and the reset gate bias is set so the gate is effectively
            inactive (near 1.0). This is enforced persistently via gradient
            masking and a forward pre-hook, so constraints hold across forward
            passes, optimizer steps, and epochs.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
        norm: "str | None" = None,
        simple_gru_mode: bool = False,
        # Initialization knobs
        weight_init: "str | None" = None,
        recurrent_weight_init: "str | None" = None,
        bias_init: "str | None" = None,
        recurrent_weight_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Build stacked single-layer GRU cells to allow norms between layers
        cells = []
        for layer_idx in range(n_layers):
            input_size = in_dim if layer_idx == 0 else hidden_dim
            cells.append(
                nn.GRU(
                    input_size=input_size,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    dropout=0.0,
                )
            )
        self.layers = nn.ModuleList(cells)

        # Optional normalization following each layer
        self.norms = nn.ModuleList()
        norm_choice = (norm or "none").lower()
        for _ in range(n_layers):
            if norm_choice in ("layernorm", "layer", "ln"):
                self.norms.append(nn.LayerNorm(hidden_dim))
            elif norm_choice in ("rmsnorm", "rms"):
                rms_cls = getattr(nn, "RMSNorm", None)
                if rms_cls is None:
                    raise ImportError(
                        "Requested norm='rmsnorm' but torch.nn.RMSNorm is not available in this PyTorch version."
                    )
                self.norms.append(rms_cls(hidden_dim))  # type: ignore[call-arg]
            elif norm_choice in ("batchnorm", "batch", "bn"):
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.norms.append(nn.Identity())

        self.drop = LockedDropout(dropout)

        # Optional simplification: disable the reset gate persistently.
        self.simple_gru_mode = bool(simple_gru_mode)
        if self.simple_gru_mode:
            self._simple_gru_reset_bias_value = 10.0  # sigmoid(10) ~= 0.99995
            self._simple_gru_weight_masks = {}
            self._simple_gru_bias_masks = {}
            self._simple_gru_bias_add = {}
            self._register_simple_gru_constraints()
            # Enforce once at init so state_dict reflects the constraints immediately.
            self._enforce_simple_gru_constraints()

        # Parameter initialization (applied after cells are created, before training)
        self._initialize_parameters(
            weight_init=weight_init,
            recurrent_weight_init=recurrent_weight_init,
            bias_init=bias_init,
            recurrent_weight_scale=float(recurrent_weight_scale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for cell, norm in zip(self.layers, self.norms):
            h, _ = cell(h)
            h = self._apply_norm(h, norm)
        return self.drop(h)

    # ---- Simple GRU mode implementation ----
    def _apply_norm(self, x: torch.Tensor, norm_layer: nn.Module) -> torch.Tensor:
        # x: [B, T, H]
        if isinstance(norm_layer, nn.BatchNorm1d):
            B, T, H = x.shape
            x2 = x.contiguous().view(B * T, H)
            x2 = norm_layer(x2)
            return x2.view(B, T, H)
        return norm_layer(x)

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
            # leave as-is for unknown
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
            # default to zeros if unknown
            nn_init.zeros_(tensor)

    def _initialize_parameters(
        self,
        *,
        weight_init: str | None,
        recurrent_weight_init: str | None,
        bias_init: str | None,
        recurrent_weight_scale: float,
    ) -> None:
        # Defaults
        w_init = weight_init or "xavier_uniform"
        hh_init = recurrent_weight_init or w_init
        b_init = bias_init or "zeros"
        scale = float(recurrent_weight_scale)
        for cell in self.layers:
            for name, param in cell.named_parameters():
                if name.startswith("weight_ih_"):
                    self._init_matrix(param.data, w_init)
                elif name.startswith("weight_hh_"):
                    self._init_matrix(param.data, hh_init)
                    if scale != 1.0:
                        param.data.mul_(scale)
                elif name.startswith("bias_"):
                    self._init_bias(param.data, b_init)

    def _register_simple_gru_constraints(self) -> None:
        # Build masks and gradient hooks so the reset gate remains disabled during training.
        self._simple_gru_weight_constraints = []  # list[(param, mask)]
        self._simple_gru_bias_constraints = []  # list[(param, mask, add_vec)]

        for cell in self.layers:
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
                    # Keep gradients only for non-reset entries.
                    mask = torch.ones_like(param.data)
                    mask[:hidden_size] = 0

                    add_vec = torch.zeros_like(param.data)
                    if name.startswith("bias_ih_"):
                        # Bias the reset gate towards 1: sigmoid(b_ir + b_hr) -> sigmoid(10 + 0)
                        add_vec[:hidden_size] = self._simple_gru_reset_bias_value
                    else:  # bias_hh_*
                        add_vec[:hidden_size] = 0.0

                    self._simple_gru_bias_constraints.append((param, mask, add_vec))

                    def _bias_grad_mask_hook(grad, m=mask):
                        return grad * m.to(grad.device)

                    param.register_hook(_bias_grad_mask_hook)

        # Forward pre-hook on the whole backbone to re-apply constraints each pass
        def _pre_hook(_module, _inputs):
            self._enforce_simple_gru_constraints()

        self._simple_gru_pre_hook_handle = self.register_forward_pre_hook(_pre_hook)

    @torch.no_grad()
    def _enforce_simple_gru_constraints(self) -> None:
        if not self.simple_gru_mode:
            return
        # Apply weight masks
        for param, mask in self._simple_gru_weight_constraints:
            param.data.mul_(mask.to(param.data.device))
        # Apply bias masks and additions
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
            # Only non-masked elements count
            total += int((mask != 0).sum().item())
        for param, mask, _add in self._simple_gru_bias_constraints:
            total += int((mask != 0).sum().item())
        # Plus all other parameters not part of the constrained sets
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


__all__ = [
    "GRUBackbone",
]


