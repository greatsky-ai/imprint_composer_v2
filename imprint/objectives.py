# imprint/objectives.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

TargetsSpec = Tuple[Any, ...]


class Targets:
    """
    Helper constructors for target specifications used by Objectives.
    """

    @staticmethod
    def batch_key(key: str) -> TargetsSpec:
        return ("batch_key", key)

    @staticmethod
    def shifted_input(source: "Module", shift: int) -> TargetsSpec:
        return ("shifted_input", source.name, shift)

    @staticmethod
    def port_drive(module: "Module", port: str) -> TargetsSpec:
        return ("port_drive", module.name, port)


def _ensure_time_alignment(
    tensor: torch.Tensor,
    target_time: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Ensure tensor has the desired time dimension, padding or truncating if needed.
    """
    if tensor.dim() < 3:
        # Expand missing dims: [B] -> [B, 1, 1], [B, D] -> [B, 1, D]
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(1).unsqueeze(2)
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(1)
    current_time = tensor.shape[1]
    if current_time == target_time:
        return tensor
    if current_time == 1:
        return tensor.expand(tensor.shape[0], target_time, *tensor.shape[2:])
    if current_time > target_time:
        return tensor[:, :target_time, ...]
    pad = target_time - current_time
    zeros = torch.zeros(
        tensor.shape[0],
        pad,
        *tensor.shape[2:],
        device=device,
        dtype=tensor.dtype,
    )
    return torch.cat([tensor, zeros], dim=1)


def _materialize_target_tensor(
    spec: TargetsSpec,
    batch: Dict[str, torch.Tensor],
    modules: Dict[str, "Module"],
) -> torch.Tensor:
    kind = spec[0]
    if kind == "batch_key":
        key = spec[1]
        if key not in batch:
            raise KeyError(f"Batch is missing key {key!r} required for objective.")
        return batch[key]
    if kind == "shifted_input":
        module_name, shift = spec[1], int(spec[2])
        if module_name not in modules:
            raise KeyError(f"Unknown module {module_name!r} in shifted_input target.")
        source_module = modules[module_name]
        if "out" not in source_module.state.output:
            raise ValueError(f"Module {module_name} has no recorded 'out' output.")
        tensor = source_module.state.output["out"]
        if shift > 0:
            tensor = tensor[:, shift:, ...]
        elif shift < 0:
            tensor = tensor[:, :shift, ...]
        if tensor.shape[1] == 0:
            raise ValueError("Shifted input target produced an empty sequence.")
        return tensor
    if kind == "port_drive":
        module_name, port = spec[1], spec[2]
        if module_name not in modules:
            raise KeyError(f"Unknown module {module_name!r} in port_drive target.")
        module = modules[module_name]
        if port not in module.state.input_drive:
            raise ValueError(f"Module {module_name} has no recorded drive for port {port!r}.")
        return module.state.input_drive[port]
    raise ValueError(f"Unsupported target spec type {kind!r}.")


def _align_trailing_dims(
    tensor: torch.Tensor,
    reference_shape: Sequence[int],
) -> torch.Tensor:
    while tensor.dim() < len(reference_shape):
        tensor = tensor.unsqueeze(-1)
    trailing_src = tensor.shape[2:]
    trailing_dst = reference_shape[2:]
    if trailing_src == trailing_dst:
        return tensor
    if all(dim == 1 for dim in trailing_src):
        expanded = tensor.expand(tensor.shape[0], tensor.shape[1], *trailing_dst)
        return expanded
    raise ValueError(
        f"Cannot align target shape {tensor.shape} to required shape {tuple(reference_shape)}."
    )


def _align_to_reference(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    aligned = tensor
    if aligned.dim() == reference.dim() - 1:
        aligned = aligned.unsqueeze(1)
    if aligned.dim() < 3 and reference.dim() >= 3:
        aligned = aligned.unsqueeze(-1)
    aligned = aligned.to(device=reference.device)
    aligned = _ensure_time_alignment(aligned, reference.shape[1], reference.device)
    return _align_trailing_dims(aligned, reference.shape)


def resolve_target(
    spec: TargetsSpec,
    batch: Dict[str, torch.Tensor],
    modules: Dict[str, "Module"],
    *,
    align_with: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Resolve a target specification into a tensor aligned with align_with.
    """
    tensor = _materialize_target_tensor(spec, batch, modules)
    if align_with is not None:
        tensor = _align_to_reference(tensor, align_with)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _zero_scalar(module: "Module") -> torch.Tensor:
    param = next(module.parameters(), None)
    device = param.device if param is not None else torch.device("cpu")
    return torch.zeros((), device=device, dtype=torch.float32)


def _activity_penalty(tensor: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "activity_l1":
        return tensor.abs().mean()
    return tensor.square().mean()


def _parameter_penalty(module: "Module", tag: str, kind: str) -> torch.Tensor:
    total: Optional[torch.Tensor] = None
    count = 0
    for param in module.iter_parameters_by_tag(tag):  # type: ignore[attr-defined]
        contrib = param.abs().sum() if kind == "params_l1" else (param.square()).sum()
        total = contrib if total is None else total + contrib
        count += param.numel()
    if total is None or count == 0:
        return _zero_scalar(module)
    return total / count


def _cross_entropy_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.dim() == pred.dim():
        indices = target.argmax(dim=-1)
    else:
        indices = target.squeeze(-1)
    while pred.dim() > 2 and indices.dim() < pred.dim() - 1:
        indices = indices.unsqueeze(-1)
    if pred.dim() == 3:
        B, T, C = pred.shape
        return F.cross_entropy(
            pred.view(B * T, C),
            indices.reshape(B * T).long(),
        )
    return F.cross_entropy(pred, indices.long())


class Objectives:
    """
    Container for per-module training objectives.
    """

    def __init__(self, module: "Module") -> None:
        self.module = module
        self._terms: List[Dict[str, Any]] = []

    def add(self, fn: Any, weight: float = 1.0) -> None:
        self._terms.append({"kind": "callable", "fn": fn, "weight": float(weight)})

    def mse(
        self,
        on: str,
        target: TargetsSpec,
        weight: float = 1.0,
    ) -> None:
        self._terms.append(
            {"kind": "mse", "on": on, "target": target, "weight": float(weight)}
        )

    def ce(
        self,
        on: str,
        target: TargetsSpec,
        weight: float = 1.0,
    ) -> None:
        self._terms.append(
            {"kind": "ce", "on": on, "target": target, "weight": float(weight)}
        )

    def activity_l1(self, on: str, weight: float = 1.0) -> None:
        self._terms.append({"kind": "activity_l1", "on": on, "weight": float(weight)})

    def activity_l2(self, on: str, weight: float = 1.0) -> None:
        self._terms.append({"kind": "activity_l2", "on": on, "weight": float(weight)})

    # --- Parameter regularization (tag-based) ---
    def params_l2(self, tag: str, weight: float = 1.0) -> None:
        """
        L2 regularization over parameters selected by a semantic tag.
        Supported tags: 'proto_params', 'recurrent_params', 'all_params'
        """
        self._terms.append({"kind": "params_l2", "tag": tag, "weight": float(weight)})

    def params_l1(self, tag: str, weight: float = 1.0) -> None:
        """
        L1 regularization over parameters selected by a semantic tag.
        Supported tags: 'proto_params', 'recurrent_params', 'all_params'
        """
        self._terms.append({"kind": "params_l1", "tag": tag, "weight": float(weight)})

    def _port_tensor(self, port: str) -> torch.Tensor:
        if port not in self.module.state.output:
            raise ValueError(f"Module {self.module.name} has no recorded output for port {port!r}.")
        return self.module.state.output[port]

    def compute(self, batch: Dict[str, torch.Tensor], modules: Dict[str, Module]) -> torch.Tensor:
        if not self._terms:
            return _zero_scalar(self.module)

        losses: List[torch.Tensor] = []
        for term in self._terms:
            kind = term["kind"]
            weight = term.get("weight", 1.0)
            if kind == "callable":
                loss = term["fn"]()
                losses.append(weight * loss)
                continue

            if kind in {"activity_l1", "activity_l2"}:
                tensor = self._port_tensor(term["on"])
                losses.append(weight * _activity_penalty(tensor, kind))
                continue

            if kind in {"mse", "ce"}:
                pred = self._port_tensor(term["on"])
                target = resolve_target(
                    term["target"],
                    batch,
                    modules,
                    align_with=pred,
                    dtype=torch.float32 if kind == "mse" else None,
                )
                if kind == "mse":
                    losses.append(weight * F.mse_loss(pred, target))
                else:
                    losses.append(weight * _cross_entropy_loss(pred, target))
                continue

            if kind in {"params_l1", "params_l2"}:
                losses.append(weight * _parameter_penalty(self.module, term["tag"], kind))
                continue

            raise ValueError(f"Unsupported objective kind: {kind}")

        return torch.stack(losses).sum()
