from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch

from .core import Graph
from .data_helper import SequenceDataset


def _train_epoch(
    graph: Graph,
    dataset: SequenceDataset,
    optimizer: Optional[torch.optim.Optimizer],
    *,
    epoch: int,
    log_every: int,
    lr: float,
    loss_fn: Optional[Callable[[Graph, Dict[str, torch.Tensor]], torch.Tensor]] = None,
    metric_fn: Optional[Callable[[Graph, Dict[str, torch.Tensor]], torch.Tensor]] = None,
    grad_clip: Optional[float] = None,
    use_adamw: bool = False,
    weight_decay: float = 0.0,
    betas: Optional[Tuple[float, float]] = None,
) -> Tuple[float, Optional[float], torch.optim.Optimizer]:
    total_loss = 0.0
    total_metric = 0.0
    steps = 0
    for step, batch in enumerate(dataset.iter_batches(shuffle=True), start=1):
        # Lazily construct parameters/optimizer on first batch via bind/rollout.
        graph.rollout(batch)
        if optimizer is None:
            if use_adamw:
                optimizer = torch.optim.AdamW(
                    graph.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=betas or (0.9, 0.999),
                )
            else:
                optimizer = torch.optim.Adam(
                    graph.parameters(),
                    lr=lr,
                    betas=betas or (0.9, 0.999),
                )
        optimizer.zero_grad()
        loss = loss_fn(graph, batch) if loss_fn is not None else graph.loss()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(graph.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        steps += 1
        if metric_fn is not None:
            with torch.no_grad():
                metric = metric_fn(graph, batch)
                total_metric += float(metric.item())

        if step % log_every == 0 or step == dataset.batches_per_epoch:
            print(
                f"[epoch {epoch}] step {step}/{dataset.batches_per_epoch} "
                f"loss={loss.item():.4f}"
            )

    assert optimizer is not None
    avg_metric = (total_metric / steps) if metric_fn is not None and steps > 0 else None
    return total_loss / dataset.batches_per_epoch, avg_metric, optimizer


@torch.no_grad()
def _eval_epoch(
    graph: Graph,
    dataset: SequenceDataset,
    *,
    loss_fn: Optional[Callable[[Graph, Dict[str, torch.Tensor]], torch.Tensor]] = None,
) -> float:
    total = 0.0
    for batch in dataset.iter_batches(shuffle=False):
        graph.rollout(batch)
        loss = loss_fn(graph, batch) if loss_fn is not None else graph.loss()
        total += loss.item()
    return total / dataset.batches_per_epoch


def train_graph(
    graph: Graph,
    dataset: SequenceDataset,
    *,
    epochs: int,
    lr: float,
    log_every: int,
    seed: Optional[int] = None,
    loss_fn: Optional[Callable[[Graph, Dict[str, torch.Tensor]], torch.Tensor]] = None,
    metric_fn: Optional[Callable[[Graph, Dict[str, torch.Tensor]], torch.Tensor]] = None,
    grad_clip: Optional[float] = None,
    use_adamw: bool = False,
    weight_decay: float = 0.0,
    betas: Optional[Tuple[float, float]] = None,
    # Optional validation
    val_dataset: Optional[SequenceDataset] = None,
    val_every: int = 1,
    val_loss_fn: Optional[Callable[[Graph, Dict[str, torch.Tensor]], torch.Tensor]] = None,
    val_metric_fn: Optional[Callable[[Graph, Dict[str, torch.Tensor]], torch.Tensor]] = None,
) -> List[float]:
    """
    Train a graph on a SequenceDataset using a simple SGD loop.

    Returns a list of average loss values per epoch.
    """
    if seed is not None:
        torch.manual_seed(seed)
    optimizer: Optional[torch.optim.Optimizer] = None
    history: List[float] = []
    for epoch in range(1, epochs + 1):
        avg_loss, avg_metric, optimizer = _train_epoch(
            graph,
            dataset,
            optimizer,
            epoch=epoch,
            log_every=log_every,
            lr=lr,
            loss_fn=loss_fn,
            metric_fn=metric_fn,
            grad_clip=grad_clip,
            use_adamw=use_adamw,
            weight_decay=weight_decay,
            betas=betas,
        )
        if avg_metric is not None:
            print(f"Epoch {epoch} average loss: {avg_loss:.4f} acc={avg_metric:.4f}")
        else:
            print(f"Epoch {epoch} average loss: {avg_loss:.4f}")
        history.append(avg_loss)
        if val_dataset is not None and val_every > 0 and (epoch % val_every) == 0:
            # Default to training loss/metric if not explicitly overridden.
            eff_val_loss_fn = val_loss_fn or loss_fn
            eff_val_metric_fn = val_metric_fn or metric_fn
            val_loss = _eval_epoch(graph, val_dataset, loss_fn=eff_val_loss_fn)
            if eff_val_metric_fn is not None:
                val_acc = _eval_epoch(graph, val_dataset, loss_fn=eff_val_metric_fn)
                print(f"[val after epoch {epoch}] avg_loss={val_loss:.4f} acc={val_acc:.4f}")
            else:
                print(f"[val after epoch {epoch}] avg_loss={val_loss:.4f}")
    return history


