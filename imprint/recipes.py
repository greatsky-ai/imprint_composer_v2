from __future__ import annotations

from typing import Callable

import torch

from .core import Graph
from .objectives import Targets
from .data_helper import SequenceDataset


def prepare_seq2static_classification(
    graph: Graph,
    dataset: SequenceDataset,
    *,
    head_name: str = "head",
    label_key: str = "y",
    emit_once: bool = False,
) -> None:
    """
    Configure a graph for sequence-to-static classification:
      - Optionally set the head to emit once per sequence (off by default).
      - Register CE objective on the head's 'out' port against batch[label_key].
    """
    head = graph.modules[head_name]
    if emit_once:
        head.schedule.emit_every = dataset.seq_len
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


