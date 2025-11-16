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
    grad_clip: Optional[float] = None,
    use_adamw: bool = False,
    weight_decay: float = 0.0,
    betas: Optional[Tuple[float, float]] = None,
) -> Tuple[float, torch.optim.Optimizer]:
    total_loss = 0.0
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

        if step % log_every == 0 or step == dataset.batches_per_epoch:
            print(
                f"[epoch {epoch}] step {step}/{dataset.batches_per_epoch} "
                f"loss={loss.item():.4f}"
            )

    assert optimizer is not None
    return total_loss / dataset.batches_per_epoch, optimizer


def train_graph(
    graph: Graph,
    dataset: SequenceDataset,
    *,
    epochs: int,
    lr: float,
    log_every: int,
    seed: Optional[int] = None,
    loss_fn: Optional[Callable[[Graph, Dict[str, torch.Tensor]], torch.Tensor]] = None,
    grad_clip: Optional[float] = None,
    use_adamw: bool = False,
    weight_decay: float = 0.0,
    betas: Optional[Tuple[float, float]] = None,
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
        avg_loss, optimizer = _train_epoch(
            graph,
            dataset,
            optimizer,
            epoch=epoch,
            log_every=log_every,
            lr=lr,
            loss_fn=loss_fn,
            grad_clip=grad_clip,
            use_adamw=use_adamw,
            weight_decay=weight_decay,
            betas=betas,
        )
        print(f"Epoch {epoch} average loss: {avg_loss:.4f}")
        history.append(avg_loss)
    return history


