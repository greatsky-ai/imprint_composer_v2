from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import torch

from .core import Auto, Graph, InPort, Module, PortRef, Ports
from .objectives import Targets
from .data_helper import SequenceDataset, load_demo_dataset
from . import protos


def prepare_seq2static_classification(
    graph: Graph,
    dataset: SequenceDataset,
    *,
    head_name: str = "head",
    label_key: str = "y",
    emit_once: bool = False,
    register_objective: bool = True,
) -> None:
    """
    Configure a graph for sequence-to-static classification:
      - Optionally set the head to emit once per sequence (off by default).
      - Optionally register CE objective on the head's 'out' port against batch[label_key].
    """
    head = graph.modules[head_name]
    if emit_once:
        head.schedule.emit_every = dataset.seq_len
    if register_objective:
        head.objectives.ce(on="out", target=Targets.batch_key(label_key))


def last_step_ce_loss(
    *,
    head_name: str = "head",
    label_key: str = "y",
) -> Callable[[Graph, dict], torch.Tensor]:
    """
    Cross-entropy on the final timestep logits of the given head vs batch[label_key].
    """
    def _loss(graph: Graph, batch: dict) -> torch.Tensor:
        head = graph.modules[head_name]
        logits = head.state.output["out"]  # [B, T, C] or [B, C]
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        y = batch[label_key]
        if y.dim() > 1:
            y = y.squeeze(-1)
        return torch.nn.functional.cross_entropy(logits, y.long())
    return _loss

def last_step_accuracy(
    *,
    head_name: str = "head",
    label_key: str = "y",
) -> Callable[[Graph, dict], torch.Tensor]:
    """
    Classification accuracy on the final timestep logits of the given head vs batch[label_key].
    Returns a scalar tensor accuracy in [0, 1].
    """
    def _acc(graph: Graph, batch: dict) -> torch.Tensor:
        head = graph.modules[head_name]
        logits = head.state.output["out"]
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        pred = logits.argmax(dim=-1)
        y = batch[label_key]
        if y.dim() > 1:
            y = y.squeeze(-1)
        correct = (pred == y.long()).float().mean()
        return correct
    return _acc


def infer_num_classes(
    dataset: SequenceDataset,
    *,
    override: Optional[int] = None,
) -> int:
    """
    Infer number of classes for classification from a dataset once:
      - Prefer explicit override when provided.
      - Else use dataset.num_classes when available.
      - Else compute 1 + max(labels) if labels exist.
      - Else fall back to dataset.target_dim.
    """
    if override is not None:
        return int(override)
    if dataset.num_classes is not None:
        return int(dataset.num_classes)
    labels = getattr(dataset, "labels", None)
    if labels is not None:
        return int(labels.max().item()) + 1
    return int(dataset.target_dim)


def detect_task_mode(
    dataset: SequenceDataset,
    *,
    num_classes_override: Optional[int] = None,
) -> Tuple[bool, Optional[int]]:
    """
    Determine whether classification labels are available for seq2static heads.
    Returns (is_classification, num_classes or None).
    """
    try:
        num_classes = infer_num_classes(dataset, override=num_classes_override)
    except Exception:
        return False, None
    if int(num_classes) <= 0:
        return False, None
    return True, int(num_classes)


def load_train_val_splits(
    data_cfg: Mapping[str, Any],
    *,
    train_split: str = "train",
    val_split: Optional[str] = "val",
) -> Tuple[SequenceDataset, Optional[SequenceDataset]]:
    """
    Convenience for demos: build train + optional val datasets from a shared config.
    """
    data_kwargs = dict(data_cfg)
    train_ds = load_demo_dataset(split=train_split, **data_kwargs)
    val_ds = None
    if val_split is not None:
        val_ds = load_demo_dataset(split=val_split, **data_kwargs)
    return train_ds, val_ds


def trainer_kwargs_from_config(
    cfg: Mapping[str, Any],
    *,
    val_dataset: Optional[SequenceDataset] = None,
) -> dict:
    """
    Extract the standard train_graph kwargs from a demo CONFIG dict.
    """
    required = ("epochs", "lr", "log_every")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise KeyError(f"Trainer config missing required keys: {', '.join(missing)}")
    train_kwargs = {key: cfg[key] for key in required}
    optional = ("seed", "grad_clip", "use_adamw", "weight_decay", "betas", "val_every")
    for key in optional:
        if key in cfg:
            train_kwargs[key] = cfg[key]
    if val_dataset is not None:
        train_kwargs["val_dataset"] = val_dataset
    return train_kwargs


def combined_graph_and_ce_loss(
    *,
    head_name: str,
    label_key: str = "y",
    weight: float = 1.0,
) -> Callable[[Graph, dict], torch.Tensor]:
    """
    Return a loss_fn that sums the graph-local objectives with a final-step CE term.
    """
    ce_loss = last_step_ce_loss(head_name=head_name, label_key=label_key)

    def _loss(graph: Graph, batch: dict) -> torch.Tensor:
        return graph.loss() + weight * ce_loss(graph, batch)

    def _ce_component(graph: Graph, batch: dict) -> torch.Tensor:
        return weight * ce_loss(graph, batch)

    _loss.components = {f"{head_name}:ce": _ce_component}  # type: ignore[attr-defined]
    return _loss


def attach_task_head(
    graph: Graph,
    *,
    dataset: SequenceDataset,
    config: Mapping[str, Any],
    feature_refs: Sequence[PortRef],
    extra_refs: Sequence[Tuple[PortRef, bool]] = (),
) -> Optional[Module]:
    """
    Attach an auxiliary recurrent block + MLP readout that consumes arbitrary PortRefs.
    """
    if not config.get("enabled", False):
        return None
    if not feature_refs and not extra_refs:
        return None

    aux_name = str(config.get("aux_name", "pc_aux"))
    head_name = str(config.get("head_name", "task_head"))
    aux_hidden = int(config.get("aux_hidden", 160))
    aux_layers = int(config.get("aux_layers", 1))
    aux_out_dim = config.get("aux_out_dim", None)
    head_widths = config.get("head_widths", [Auto])
    stop_grad = bool(config.get("stop_grad", True))
    classification_mode, num_classes = detect_task_mode(
        dataset, num_classes_override=config.get("num_classes")
    )

    aux = Module(
        name=aux_name,
        proto=protos.GRUStack(
            hidden=aux_hidden,
            layers=aux_layers,
            layernorm=True,
            out_dim=(None if aux_out_dim is None else int(aux_out_dim)),
        ),
        ports=Ports(
            in_=InPort(size=Auto, combine="concat"),
            out_default=Auto,
        ),
    )
    head = Module(
        name=head_name,
        proto=protos.MLP(widths=head_widths),
        ports=Ports(
            in_default=Auto,
            out_default=(num_classes if classification_mode else Auto),
        ),
    )
    graph.add(aux, head)

    def _connect(ref: PortRef, sg: bool) -> None:
        edge = graph.connect(ref, aux["in"])
        edge.set_stop_grad(sg)

    for ref in feature_refs:
        _connect(ref, stop_grad)
    for ref, sg in extra_refs:
        _connect(ref, sg)

    graph.connect(aux["out"], head["in"])
    return head


