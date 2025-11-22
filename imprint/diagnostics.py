from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.hooks import RemovableHandle

from .core import Edge, Graph, Module
from .data_helper import SequenceDataset


@dataclass
class StatRecord:
    name: str
    l2: float
    max_abs: float
    mean_abs: float
    zero_frac: float


@dataclass
class GradientSummary:
    modules: List[StatRecord]
    edges: List[StatRecord]
    ports: List[StatRecord]
    stop_grad_edges: List[str]

    def to_text(self, top_k: Optional[int] = None) -> str:
        sections: List[str] = []

        def _fmt_section(title: str, rows: Sequence[StatRecord]) -> Optional[str]:
            if not rows:
                return None
            lines = [f"{title} gradients:"]
            limit = rows if top_k is None else rows[:top_k]
            for rec in limit:
                lines.append(
                    f"  {rec.name:<30} |l2|={rec.l2:.4e} "
                    f"|max|={rec.max_abs:.4e} mean|g|={rec.mean_abs:.4e} "
                    f"zero%={rec.zero_frac * 100:5.2f}"
                )
            return "\n".join(lines)

        for label, rows in (
            ("Module", self.modules),
            ("Edge", self.edges),
            ("Port", self.ports),
        ):
            block = _fmt_section(label, rows)
            if block:
                sections.append(block)

        if self.stop_grad_edges:
            sections.append(
                "Stop-grad edges: "
                + ", ".join(sorted(self.stop_grad_edges))
            )

        return "\n".join(sections)


class _StatBucket:
    __slots__ = ("entries", "l2_sum", "abs_sum", "max_abs", "zero_count", "elem_count")

    def __init__(self) -> None:
        self.entries = 0
        self.l2_sum = 0.0
        self.abs_sum = 0.0
        self.max_abs = 0.0
        self.zero_count = 0
        self.elem_count = 0

    def add(self, tensor: torch.Tensor) -> None:
        if tensor is None:
            return
        data = tensor.detach()
        if data.numel() == 0:
            return
        abs_val = data.abs()
        self.entries += 1
        self.l2_sum += float(data.norm().item())
        self.abs_sum += float(abs_val.sum().item())
        self.max_abs = max(self.max_abs, float(abs_val.max().item()))
        zeros = (abs_val <= 1e-9).sum().item()
        self.zero_count += int(zeros)
        self.elem_count += data.numel()

    def to_record(self, name: str) -> StatRecord:
        if self.entries == 0 or self.elem_count == 0:
            return StatRecord(name=name, l2=0.0, max_abs=0.0, mean_abs=0.0, zero_frac=0.0)
        l2 = self.l2_sum / self.entries
        mean_abs = self.abs_sum / self.elem_count
        zero_frac = self.zero_count / max(1, self.elem_count)
        return StatRecord(
            name=name,
            l2=l2,
            max_abs=self.max_abs,
            mean_abs=mean_abs,
            zero_frac=zero_frac,
        )


class GradientWatcher:
    """
    Register gradient hooks on module parameters, edge projections, and optional
    proto outputs to track gradient flow during training.
    """

    def __init__(
        self,
        graph: Graph,
        *,
        track_ports: bool = True,
        track_edges: bool = True,
    ) -> None:
        self.graph = graph
        self.track_ports = track_ports
        self.track_edges = track_edges

        self._module_handles: List[RemovableHandle] = []
        self._proto_handles: List[RemovableHandle] = []
        self._edge_handles: List[RemovableHandle] = []
        self._hooked_edges: set[int] = set()

        self._module_stats: Dict[str, _StatBucket] = {}
        self._edge_stats: Dict[str, _StatBucket] = {}
        self._port_stats: Dict[str, _StatBucket] = {}

        self.graph.register_event_listener(self._on_event)
        self._attach_module_param_hooks()
        if self.track_ports:
            self._attach_proto_hooks()
        if self.track_edges:
            self._ensure_edge_hooks()

    def close(self) -> None:
        for handle in self._module_handles:
            handle.remove()
        for handle in self._proto_handles:
            handle.remove()
        for handle in self._edge_handles:
            handle.remove()
        self.graph.unregister_event_listener(self._on_event)
        self._module_handles.clear()
        self._proto_handles.clear()
        self._edge_handles.clear()
        self._hooked_edges.clear()

    def _on_event(self, payload: Dict[str, Any]) -> None:
        if payload.get("event") == "rollout_start" and self.track_edges:
            self._ensure_edge_hooks()

    def _attach_module_param_hooks(self) -> None:
        for module in self.graph.modules.values():
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                handle = param.register_hook(
                    self._make_param_hook(self._module_stats, module.name)
                )
                self._module_handles.append(handle)

    def _attach_proto_hooks(self) -> None:
        for module in self.graph.modules.values():
            proto = module.proto
            handle = proto.register_full_backward_hook(
                self._make_proto_hook(module.name)
            )
            self._proto_handles.append(handle)

    def _ensure_edge_hooks(self) -> None:
        for edge in self.graph.edges:
            if edge.proj is None:
                continue
            edge_id = id(edge)
            if edge_id in self._hooked_edges:
                continue
            prefix = f"{edge.src.module.name}->{edge.dst.module.name}"
            for name, param in edge.proj.named_parameters():
                if not param.requires_grad:
                    continue
                handle = param.register_hook(
                    self._make_param_hook(self._edge_stats, f"{prefix}.{name}")
                )
                self._edge_handles.append(handle)
            self._hooked_edges.add(edge_id)

    def _make_param_hook(
        self,
        store: Dict[str, _StatBucket],
        label: str,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        def _hook(grad: torch.Tensor) -> torch.Tensor:
            bucket = store.setdefault(label, _StatBucket())
            bucket.add(grad)
            return grad

        return _hook

    def _make_proto_hook(
        self,
        module_name: str,
    ) -> Callable[[torch.nn.Module, Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]], None]:
        def _hook(
            module: torch.nn.Module,
            grad_inputs: Tuple[torch.Tensor, ...],
            grad_outputs: Tuple[torch.Tensor, ...],
        ) -> None:
            del module, grad_inputs
            if not grad_outputs:
                return
            grad = grad_outputs[0]
            if grad is None:
                return
            bucket = self._port_stats.setdefault(f"{module_name}.out", _StatBucket())
            bucket.add(grad)

        return _hook

    def reset(self) -> None:
        self._module_stats.clear()
        self._edge_stats.clear()
        self._port_stats.clear()

    def pop_summary(self, *, top_k: Optional[int] = None) -> Optional[GradientSummary]:
        modules = self._consume(self._module_stats, top_k)
        edges = self._consume(self._edge_stats, top_k)
        ports = self._consume(self._port_stats, top_k)
        if not modules and not edges and not ports:
            return None
        stop_edges = [
            edge.name for edge in self.graph.edges if getattr(edge, "stop_grad", False)
        ]
        summary = GradientSummary(
            modules=modules,
            edges=edges,
            ports=ports,
            stop_grad_edges=stop_edges,
        )
        return summary

    def _consume(
        self,
        store: Dict[str, _StatBucket],
        top_k: Optional[int],
    ) -> List[StatRecord]:
        if not store:
            return []
        items = [bucket.to_record(name) for name, bucket in store.items()]
        store.clear()
        items.sort(key=lambda rec: rec.l2, reverse=True)
        if top_k is not None:
            return items[:top_k]
        return items


def plot_gradient_heatmap(
    summary: GradientSummary,
    *,
    section: str = "modules",
    metric: str = "l2",
    ax: Optional["matplotlib.axes.Axes"] = None,
) -> "matplotlib.axes.Axes":
    """
    Render a 1×N heatmap for the requested gradient metric using matplotlib.
    """
    rows = getattr(summary, section, None)
    if not rows:
        raise ValueError(f"No rows available for section {section!r}.")
    values = [getattr(row, metric) for row in rows]
    labels = [row.name for row in rows]
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for heatmap rendering.") from exc

    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, len(values)), 2))
    data = np.array([values], dtype=float)
    im = ax.imshow(data, aspect="auto", cmap="magma")
    ax.set_yticks([])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(f"{section.capitalize()} gradient {metric}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


@torch.inference_mode()
def visualize_module_output(
    graph: Graph,
    dataset: SequenceDataset,
    module_name: str,
    *,
    port: str = "out",
    mode: str = "trace",
    sample_index: Optional[int] = None,
    time_index: Optional[int] = -1,
    max_traces: int = 16,
    collapse_to_square: bool = True,
    ax: Optional["matplotlib.axes.Axes"] = None,
    save_path: str = "val_viz.png",
    title: Optional[str] = None,
) -> "matplotlib.axes.Axes":
    """
    Visualize a module's output on a random validation sample.

    Args:
        graph: Bound graph whose modules will be evaluated.
        dataset: SequenceDataset providing validation sequences.
        module_name: Name of the module to visualize.
        port: Output port to sample (default: 'out').
        mode: 'trace' to plot feature traces over time, 'frame' to render a single
            timestep as an image after reshaping to a square grid.
        sample_index: Optional explicit sequence index; defaults to a random pick.
        time_index: Optional explicit timestep; defaults to the final tick (-1).
            Negative indices count from the end (e.g., -1 → last tick).
        max_traces: Max number of feature traces to overlay (trace mode only).
        collapse_to_square: Pad/reshape feature vectors into the smallest square
            when rendering frames.
        ax: Optional matplotlib axes to draw on; when omitted, a new figure is created.
        save_path: Filepath for the rendered PNG (default 'val_viz.png').
        title: Optional custom plot title.
    Returns:
        The matplotlib Axes containing the visualization.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for visualization helpers.") from exc

    if dataset.num_sequences == 0:
        raise ValueError("Dataset is empty; cannot draw samples.")
    if module_name not in graph.modules:
        raise KeyError(f"Graph has no module named {module_name!r}.")
    module = graph.modules[module_name]

    sample_idx = sample_index
    if sample_idx is None:
        sample_idx = int(torch.randint(dataset.num_sequences, (1,), device=dataset.data.device).item())
    sample_idx = int(sample_idx) % dataset.num_sequences
    print(
        "[visualize_module_output]"
        f" module={module_name}.{port} mode={mode} sample={sample_idx}"
    )

    batch = {"x": dataset.data[sample_idx : sample_idx + 1]}
    if dataset.labels is not None:
        batch["y"] = dataset.labels[sample_idx : sample_idx + 1]

    primary_device: Optional[torch.device] = None
    for param in graph.parameters():
        primary_device = param.device
        break
    if primary_device is not None:
        batch = {key: tensor.to(primary_device) for key, tensor in batch.items()}

    graph.rollout(batch)

    if port not in module.state.output:
        raise KeyError(f"Module {module_name!r} has no recorded port {port!r}.")
    tensor = module.state.output[port]
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(1)
    if tensor.dim() != 3:
        raise ValueError(
            f"Expected module output with shape [B, T, D]; got {tuple(tensor.shape)}."
        )
    seq = tensor[0].detach().cpu()
    ticks, feat_dim = seq.shape
    if ticks == 0:
        raise RuntimeError(f"Module {module_name}.{port} emitted zero timesteps.")
    print(
        "[visualize_module_output]"
        f" output_shape=(ticks={ticks}, feat_dim={feat_dim})"
    )

    tick_idx = time_index
    if tick_idx is None:
        tick_idx = int(torch.randint(ticks, (1,)).item())
    else:
        tick_idx = int(tick_idx)
        if tick_idx < 0:
            tick_idx = ticks + tick_idx
    tick_idx = max(0, min(tick_idx, ticks - 1))
    print(
        "[visualize_module_output]"
        f" using_tick={tick_idx}"
    )

    if mode not in ("trace", "frame"):
        raise ValueError("mode must be 'trace' or 'frame'.")

    fig: Optional["matplotlib.figure.Figure"] = None
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(10, 4) if mode == "trace" else (4, 4)
        )
    else:
        fig = ax.figure

    resolved_title = title or f"{module_name}.{port} ({mode})"

    if mode == "trace":
        max_traces = max(1, min(int(max_traces), feat_dim))
        if max_traces == feat_dim:
            selected = torch.arange(feat_dim)
        else:
            selected = torch.linspace(0, feat_dim - 1, steps=max_traces).long()
        data = seq[:, selected].numpy()
        time_axis = np.arange(ticks)
        ax.plot(time_axis, data, alpha=0.8)
        ax.axvline(tick_idx, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_xlabel("tick")
        ax.set_ylabel("activation")
        ax.set_title(resolved_title + f" | sample={sample_idx}")
    else:
        vector = seq[tick_idx]
        frame = _vector_to_square(vector, collapse_to_square).numpy()
        im = ax.imshow(frame, cmap="magma", interpolation="nearest")
        ax.set_title(resolved_title + f" | sample={sample_idx} tick={tick_idx}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if fig is None:
        fig = ax.figure
    fig.tight_layout()
    abs_path = os.path.abspath(save_path)
    fig.savefig(abs_path, dpi=150)
    print(
        "[visualize_module_output]"
        f" saved_path={abs_path}"
    )
    plt.close(fig)
    return ax


def _vector_to_square(vector: torch.Tensor, collapse: bool) -> torch.Tensor:
    length = vector.numel()
    side = math.isqrt(length)
    if side * side != length:
        if not collapse:
            raise ValueError(
                f"Feature dim {length} is not a perfect square; set collapse_to_square=True."
            )
        side = int(math.ceil(math.sqrt(length)))
        padded = torch.zeros(side * side, dtype=vector.dtype)
        padded[:length] = vector
        vector = padded
    return vector.view(side, side)

