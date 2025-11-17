from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from .core import Graph
from .data_helper import SequenceDataset

Batch = Dict[str, torch.Tensor]
LossFn = Optional[Callable[[Graph, Batch], torch.Tensor]]


@dataclass(frozen=True)
class TrainLoopConfig:
    epochs: int
    lr: float
    log_every: int
    grad_clip: Optional[float] = None
    use_adamw: bool = False
    weight_decay: float = 0.0
    betas: Optional[Tuple[float, float]] = None
    val_every: int = 1


@dataclass
class EpochStats:
    avg_loss: float
    final_loss: float
    avg_metric: Optional[float]


class Trainer:
    def __init__(
        self,
        graph: Graph,
        dataset: SequenceDataset,
        config: TrainLoopConfig,
        *,
        loss_fn: LossFn = None,
        metric_fn: LossFn = None,
        val_dataset: Optional[SequenceDataset] = None,
        val_loss_fn: LossFn = None,
        val_metric_fn: LossFn = None,
    ) -> None:
        self.graph = graph
        self.dataset = dataset
        self.config = config
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.val_dataset = val_dataset
        self.val_loss_fn = val_loss_fn
        self.val_metric_fn = val_metric_fn
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def run(self, *, seed: Optional[int] = None) -> List[float]:
        if seed is not None:
            torch.manual_seed(seed)
        history: List[float] = []
        for epoch in range(1, self.config.epochs + 1):
            stats = self._train_epoch(epoch)
            history.append(stats.avg_loss)
            self._log_epoch(epoch, stats)
            if (
                self.val_dataset is not None
                and self.config.val_every > 0
                and (epoch % self.config.val_every) == 0
            ):
                self._log_validation(epoch)
        return history

    def _train_epoch(self, epoch: int) -> EpochStats:
        total_loss = 0.0
        total_metric = 0.0
        steps = 0
        last_loss_value = 0.0

        for step, batch in enumerate(self.dataset.iter_batches(shuffle=True), start=1):
            self.graph.rollout(batch)
            optimizer = self._ensure_optimizer()
            optimizer.zero_grad()
            loss = self._compute_loss(batch)
            loss.backward()
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.config.grad_clip)
            optimizer.step()

            last_loss_value = float(loss.item())
            total_loss += last_loss_value
            steps += 1

            metric_value = self._compute_metric(batch)
            if metric_value is not None:
                total_metric += metric_value

            if step % self.config.log_every == 0 or step == self.dataset.batches_per_epoch:
                print(
                    f"[epoch {epoch}] step {step}/{self.dataset.batches_per_epoch} "
                    f"loss={last_loss_value:.4f}"
                )

        avg_metric = (total_metric / steps) if (self.metric_fn is not None and steps > 0) else None
        avg_loss = total_loss / self.dataset.batches_per_epoch
        return EpochStats(avg_loss=avg_loss, final_loss=last_loss_value, avg_metric=avg_metric)

    def _ensure_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is not None:
            return self.optimizer
        betas = self.config.betas or (0.9, 0.999)
        if self.config.use_adamw:
            self.optimizer = torch.optim.AdamW(
                self.graph.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                betas=betas,
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.graph.parameters(),
                lr=self.config.lr,
                betas=betas,
            )
        return self.optimizer

    def _compute_loss(self, batch: Batch) -> torch.Tensor:
        return self._loss_with_objectives(self.loss_fn, batch)

    def _compute_metric(self, batch: Batch) -> Optional[float]:
        if self.metric_fn is None:
            return None
        with torch.no_grad():
            metric = self.metric_fn(self.graph, batch)
        return float(metric.item())

    def _log_epoch(self, epoch: int, stats: EpochStats) -> None:
        if stats.avg_metric is not None:
            print(f"Epoch {epoch} final loss: {stats.final_loss:.4f} acc={stats.avg_metric:.4f}")
        else:
            print(f"Epoch {epoch} final loss: {stats.final_loss:.4f}")

    def _log_validation(self, epoch: int) -> None:
        assert self.val_dataset is not None
        loss_hook = self.val_loss_fn or self.loss_fn
        metric_hook = self.val_metric_fn or self.metric_fn
        val_loss = self._evaluate_loss(self.val_dataset, loss_hook)
        if metric_hook is not None:
            val_metric = self._evaluate_metric(self.val_dataset, metric_hook)
            print(f"[val after epoch {epoch}] avg_loss={val_loss:.4f} acc={val_metric:.4f}")
        else:
            print(f"[val after epoch {epoch}] avg_loss={val_loss:.4f}")

    @torch.inference_mode()
    def _evaluate_loss(self, dataset: SequenceDataset, hook: LossFn) -> float:
        total = 0.0
        for batch in dataset.iter_batches(shuffle=False):
            self.graph.rollout(batch)
            value = self._loss_with_objectives(hook, batch)
            total += float(value.item())
        return total / dataset.batches_per_epoch

    @torch.inference_mode()
    def _evaluate_metric(self, dataset: SequenceDataset, hook: LossFn) -> float:
        total = 0.0
        for batch in dataset.iter_batches(shuffle=False):
            self.graph.rollout(batch)
            value = hook(self.graph, batch)
            total += float(value.item())
        return total / dataset.batches_per_epoch

    def _loss_with_objectives(self, hook: LossFn, batch: Batch) -> torch.Tensor:
        objective = self.graph.loss()
        if hook is None:
            return objective
        return hook(self.graph, batch) + objective


def train_graph(
    graph: Graph,
    dataset: SequenceDataset,
    *,
    epochs: int,
    lr: float,
    log_every: int,
    seed: Optional[int] = None,
    loss_fn: LossFn = None,
    metric_fn: LossFn = None,
    grad_clip: Optional[float] = None,
    use_adamw: bool = False,
    weight_decay: float = 0.0,
    betas: Optional[Tuple[float, float]] = None,
    val_dataset: Optional[SequenceDataset] = None,
    val_every: int = 1,
    val_loss_fn: LossFn = None,
    val_metric_fn: LossFn = None,
) -> List[float]:
    """
    Train a graph on a SequenceDataset using an optimizer-backed loop.

    Returns a list of average loss values per epoch.
    """
    config = TrainLoopConfig(
        epochs=epochs,
        lr=lr,
        log_every=log_every,
        grad_clip=grad_clip,
        use_adamw=use_adamw,
        weight_decay=weight_decay,
        betas=betas,
        val_every=val_every,
    )
    trainer = Trainer(
        graph,
        dataset,
        config,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        val_dataset=val_dataset,
        val_loss_fn=val_loss_fn,
        val_metric_fn=val_metric_fn,
    )
    return trainer.run(seed=seed)