from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

from .core import Graph
from . import diagnostics as imprint_diagnostics
from .data_helper import SequenceDataset

Batch = Dict[str, torch.Tensor]
LossFn = Optional[Callable[[Graph, Batch], torch.Tensor]]

if TYPE_CHECKING:
    from .diagnostics import GradientWatcher


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
        grad_monitor: Optional["GradientWatcher"] = None,
        grad_summary_top_k: Optional[int] = 5,
        viz_config: Optional[Dict[str, Any]] = None,
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
        self.grad_monitor = grad_monitor
        self.grad_summary_top_k = grad_summary_top_k
        self.viz_config = dict(viz_config) if viz_config is not None else None

    def run(self, *, seed: Optional[int] = None) -> List[float]:
        if seed is not None:
            torch.manual_seed(seed)
        history: List[float] = []
        for epoch in range(1, self.config.epochs + 1):
            stats = self._train_epoch(epoch)
            history.append(stats.avg_loss)
            self._log_epoch(epoch, stats)
            self._maybe_visualize(epoch, split="train")
            if (
                self.val_dataset is not None
                and self.config.val_every > 0
                and (epoch % self.config.val_every) == 0
            ):
                self._log_validation(epoch)
                self._maybe_visualize(epoch, split="val")
        return history

    def _train_epoch(self, epoch: int) -> EpochStats:
        total_loss = 0.0
        total_metric = 0.0
        steps = 0
        last_loss_value = 0.0
        last_by_label: Optional[Dict[str, float]] = None

        for step, batch in enumerate(self.dataset.iter_batches(shuffle=True), start=1):
            self.graph.rollout(batch)
            optimizer = self._ensure_optimizer()
            optimizer.zero_grad()

            # When using default objectives, use graph loss and collect breakdown.
            # Otherwise, compute custom loss but still collect breakdown for logging only.
            if self.loss_fn is None:
                loss_tensor, by_label_tensor, _ = self.graph.loss_breakdown()
                loss = loss_tensor
                last_by_label = {k: float(v.item()) for k, v in by_label_tensor.items()}
            else:
                loss = self._compute_loss(batch)
                with torch.no_grad():
                    _, by_label_tensor, _ = self.graph.loss_breakdown()
                    last_by_label = {k: float(v.item()) for k, v in by_label_tensor.items()}
                    # Optional: include custom loss components exposed by the loss_fn
                    components = getattr(self.loss_fn, "components", None) or getattr(self.loss_fn, "_components", None)
                    if isinstance(components, dict):
                        for name, fn in components.items():
                            try:
                                contrib = fn(self.graph, batch)
                                last_by_label[name] = float(contrib.item())
                            except Exception:
                                # Component logging is best-effort; ignore errors
                                continue

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
                if last_by_label:
                    # Sort descending by contribution for readability
                    parts = " ".join(
                        f"{k}={v:.4f}" for k, v in sorted(last_by_label.items(), key=lambda kv: -kv[1])
                    )
                    print(
                        f"[epoch {epoch}] step {step}/{self.dataset.batches_per_epoch} "
                        f"loss={last_loss_value:.4f} {parts}"
                    )
                else:
                    print(
                        f"[epoch {epoch}] step {step}/{self.dataset.batches_per_epoch} "
                        f"loss={last_loss_value:.4f}"
                    )
                self._log_gradient_summary()

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
        if self.loss_fn is None:
            return self.graph.loss()
        return self.loss_fn(self.graph, batch)

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
        val_loss = self._evaluate(self.val_dataset, loss_hook)
        if metric_hook is not None:
            val_metric = self._evaluate(self.val_dataset, metric_hook)
            print(f"[val after epoch {epoch}] avg_loss={val_loss:.4f} acc={val_metric:.4f}")
        else:
            print(f"[val after epoch {epoch}] avg_loss={val_loss:.4f}")

    @torch.inference_mode()
    def _evaluate(self, dataset: SequenceDataset, hook: LossFn) -> float:
        total = 0.0
        for batch in dataset.iter_batches(shuffle=False):
            self.graph.rollout(batch)
            value = self._invoke_hook(hook, batch)
            total += float(value.item())
        return total / dataset.batches_per_epoch

    def _invoke_hook(self, hook: LossFn, batch: Batch) -> torch.Tensor:
        if hook is None:
            return self.graph.loss()
        return hook(self.graph, batch)

    def _log_gradient_summary(self) -> None:
        if self.grad_monitor is None:
            return
        summary = self.grad_monitor.pop_summary(top_k=self.grad_summary_top_k)
        if summary is None:
            return
        text = summary.to_text()
        if not text:
            return
        for line in text.splitlines():
            print(f"    {line}")

    def _maybe_visualize(self, epoch: int, split: str) -> None:
        if not self.viz_config:
            return
        if split == "train":
            sample_dataset = self.dataset
        elif split == "val":
            if self.val_dataset is None:
                return
            sample_dataset = self.val_dataset
        else:
            return

        split_cfg = self.viz_config.get(split)
        if not split_cfg:
            return
        if isinstance(split_cfg, dict):
            entries = [split_cfg]
        elif isinstance(split_cfg, list):
            entries = split_cfg
        else:
            return

        for entry in entries:
            if not entry or not entry.get("enabled", False):
                continue
            module_name = entry.get("module")
            if not module_name:
                continue
            module_name = str(module_name)
            port = str(entry.get("port", "out"))
            mode = str(entry.get("mode", "frame"))
            save_path = entry.get("path")
            if save_path is None:
                save_dir = str(entry.get("dir", "."))
                filename = str(entry.get("filename", f"{split}_viz_epoch{epoch:03d}.png"))
                save_path = os.path.join(save_dir, filename)
            imprint_diagnostics.visualize_module_output(
                self.graph,
                sample_dataset,
                module_name=module_name,
                port=port,
                mode=mode,
                save_path=save_path,
                title=f"{module_name}.{port} ({mode}) epoch={epoch} split={split}",
            )
            # Per-epoch simple stats log for this module's 'out' port on a deterministic batch.
            try:
                first_batch = next(sample_dataset.iter_batches(shuffle=False))
                self.graph.rollout(first_batch)
                module = self.graph.modules.get(module_name)
                if module is not None and "out" in module.state.output:
                    tensor = module.state.output["out"]
                    if tensor.dim() == 2:
                        tensor = tensor.unsqueeze(1)
                    B, T, D = tensor.shape
                    overall_std = float(tensor.std().item())
                    std_over_time = float(tensor.std(dim=1).mean().item())
                    std_over_feat = float(tensor.std(dim=2).mean().item())
                    min_val = float(tensor.min().item())
                    max_val = float(tensor.max().item())
                    print(
                        f"[monitor][epoch {epoch}][{split}] {module_name}.out "
                        f"shape=[{B},{T},{D}] std_all={overall_std:.4e} "
                        f"mean(std_over_time)={std_over_time:.4e} "
                        f"mean(std_over_feat)={std_over_feat:.4e} "
                        f"min={min_val:.4e} max={max_val:.4e}"
                    )
            except Exception:
                # Best-effort monitoring; never fail the training loop.
                pass


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
    grad_monitor: Optional["GradientWatcher"] = None,
    grad_summary_top_k: Optional[int] = 5,
    viz_config: Optional[Dict[str, Any]] = None,
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
        grad_monitor=grad_monitor,
        grad_summary_top_k=grad_summary_top_k,
        viz_config=viz_config,
    )
    return trainer.run(seed=seed)