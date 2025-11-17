from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from .core import Graph
from .objectives import Targets
from .data_helper import SequenceDataset, load_demo_dataset
from .training import train_graph


@dataclass
class DemoConfig:
    """
    Helper for demos/examples to standardize dataset + training boilerplate.
    """

    seed: int = 0
    epochs: int = 5
    lr: float = 1e-3
    log_every: int = 10
    val_every: int = 1
    grad_clip: float = 1.0
    num_classes: Optional[int] = None
    train_split: str = "train"
    val_split: str = "val"
    data: Dict[str, Any] = field(default_factory=dict)

    def load_datasets(self) -> Tuple[SequenceDataset, SequenceDataset]:
        data_cfg = dict(self.data)
        train = load_demo_dataset(split=self.train_split, **data_cfg)
        val = load_demo_dataset(split=self.val_split, **data_cfg)
        return train, val

    def training_kwargs(self) -> Dict[str, Any]:
        return {
            "epochs": self.epochs,
            "lr": self.lr,
            "log_every": self.log_every,
            "seed": self.seed,
            "grad_clip": self.grad_clip,
            "val_every": self.val_every,
        }

    def infer_num_classes(self, dataset: SequenceDataset) -> int:
        return infer_num_classes(dataset, override=self.num_classes)

    def train(
        self,
        graph: Graph,
        dataset: SequenceDataset,
        *,
        loss_fn: Callable[[Graph, dict], torch.Tensor],
        metric_fn: Callable[[Graph, dict], torch.Tensor],
        val_dataset: Optional[SequenceDataset] = None,
        **extra_train_kwargs: Any,
    ) -> Any:
        if val_dataset is None:
            _, val_dataset = self.load_datasets()
        kwargs = self.training_kwargs()
        kwargs.update(extra_train_kwargs)
        return train_graph(
            graph,
            dataset,
            loss_fn=loss_fn,
            metric_fn=metric_fn,
            val_dataset=val_dataset,
            **kwargs,
        )


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


