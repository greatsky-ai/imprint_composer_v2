"""
Lightweight helpers for loading or synthesizing seq2seq demo datasets.

The goal is to keep demo scripts free of data boilerplate: a single call
returns tensors plus metadata such as feature dimensions and batches/epoch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple, Union

import torch

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    h5py = None  # type: ignore


Split = str
PathLike = Union[str, Path]


@dataclass
class SequenceDataset:
    """
    Simple container for `[N, T, D]` sequences plus optional labels/targets.
    """

    data: torch.Tensor
    labels: Optional[torch.Tensor]
    batch_size: int
    split: Split
    task_type: str
    _num_classes: Optional[int] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.data.dim() != 3:
            raise ValueError("SequenceDataset expects data shaped [N, T, D].")
        self.data = self.data.contiguous()
        if self.labels is not None:
            self.labels = self.labels.contiguous()
        self._num_classes = self._infer_num_classes()

    @property
    def num_sequences(self) -> int:
        return int(self.data.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.data.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.data.shape[2])

    @property
    def batches_per_epoch(self) -> int:
        return max(1, math.ceil(self.num_sequences / self.batch_size))

    @property
    def num_classes(self) -> Optional[int]:
        return self._num_classes

    @property
    def target_dim(self) -> int:
        """
        Best-effort guess of the logit/feature dimension required by a head.
        """
        if self.labels is None:
            return self.feature_dim
        if self._is_classification_labels():
            return self.num_classes or self.feature_dim
        if self.labels.dim() == 1:
            return 1
        return int(self.labels.shape[-1])

    def iter_batches(
        self,
        *,
        shuffle: bool = True,
        device: Optional[torch.device] = None,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Yield batches shaped for imprint graphs. Always provides ``batch['x']``.
        """
        indices = (
            torch.randperm(self.num_sequences)
            if shuffle
            else torch.arange(self.num_sequences)
        )
        for start in range(0, indices.numel(), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            batch = {"x": self.data[batch_idx]}
            if self.labels is not None:
                batch["y"] = self.labels[batch_idx]
            if device is not None:
                batch = {k: v.to(device) for k, v in batch.items()}
            yield batch

    def metadata(self) -> Dict[str, Union[str, int, None]]:
        return {
            "split": self.split,
            "task_type": self.task_type,
            "num_sequences": self.num_sequences,
            "seq_len": self.seq_len,
            "feature_dim": self.feature_dim,
            "target_dim": self.target_dim,
            "num_classes": self.num_classes,
            "batch_size": self.batch_size,
            "batches_per_epoch": self.batches_per_epoch,
        }

    def summary(self) -> str:
        meta = self.metadata()
        return (
            f"{meta['split']} split: {meta['num_sequences']} sequences × "
            f"{meta['seq_len']} ticks × {meta['feature_dim']} dims "
            f"(batch={meta['batch_size']}, steps/epoch={meta['batches_per_epoch']})"
        )

    def _infer_num_classes(self) -> Optional[int]:
        if self.labels is None:
            return None
        if not self._is_classification_labels():
            return None
        max_label = int(self.labels.max().item())
        return max_label + 1

    def _is_classification_labels(self) -> bool:
        if self.labels is None:
            return False
        return self.labels.dtype in (torch.int64, torch.int32, torch.int16)


def load_demo_dataset(
    *,
    split: Split = "train",
    batch_size: int = 16,
    path: Optional[PathLike] = None,
    config: Optional[Mapping[str, Any]] = None,
    **overrides: Any,
) -> SequenceDataset:
    """
    Load the dataset used by demo #4 (micro-stepping) or synthesize one on-demand.

    Args:
        split: Dataset split to fetch (``train``, ``val``, ``test``, or custom keys).
        batch_size: Mini-batch size used by the demo/training loop.
        path: Optional path to an HDF5 file following DATASETS.md conventions.
              If omitted, a synthetic dataset is created in-memory.
        config: Optional mapping providing default values for the loader. Supported keys
            include ``path``, ``batch_size``, ``synth_total``, ``synth_seq_len``,
            ``synth_feature_dim``, ``synth_seed``, ``synth_bounds``, and ``task_type``.
        **overrides: Keyword overrides applied last (take precedence over ``config``).
    """
    cfg = _default_micro_step_config()
    cfg.update({"split": split, "batch_size": batch_size, "path": path})
    if config is not None:
        cfg.update(dict(config))
    if overrides:
        cfg.update(overrides)

    resolved_split = str(cfg["split"]).lower()
    if not resolved_split:
        raise ValueError("split must be a non-empty string")
    batch_size = int(cfg["batch_size"])
    dataset_path = cfg.get("path")

    synth_kwargs = {
        "total": int(cfg["synth_total"]),
        "seq_len": int(cfg["synth_seq_len"]),
        "dim": int(cfg["synth_feature_dim"]),
        "seed": int(cfg["synth_seed"]),
        "bounds": dict(cfg["synth_bounds"]),
    }

    if dataset_path is None:
        data, labels = _synthesize_split(resolved_split, **synth_kwargs)
    else:
        data, labels = _load_or_create_hdf5(Path(dataset_path), resolved_split, synth_kwargs)

    return SequenceDataset(
        data=data,
        labels=labels,
        batch_size=batch_size,
        split=resolved_split,
        task_type=str(cfg["task_type"]),
    )


def scale_sequence_dataset(dataset: SequenceDataset, factor: float) -> SequenceDataset:
    """
    Scale all signals (and optional labels) in-place by ``factor``.

    Useful for emphasizing high-energy features without editing downstream graphs.
    """
    if factor == 1.0:
        return dataset
    dataset.data.mul_(factor)
    if dataset.labels is not None:
        dataset.labels.mul_(factor)
    return dataset


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_synth_bounds() -> Dict[Split, Tuple[float, float]]:
    return {"train": (0.0, 0.7), "val": (0.7, 0.85), "test": (0.85, 1.0)}


def _default_micro_step_config() -> Dict[str, Any]:
    return {
        "path": None,
        "split": "train",
        "batch_size": 16,
        "synth_total": 320,
        "synth_seq_len": 160,
        "synth_feature_dim": 64,
        "synth_seed": 7,
        "synth_bounds": _default_synth_bounds(),
        "task_type": "self_supervised_seq2seq",
    }


def _synthesize_split(
    split: Split,
    *,
    total: int,
    seq_len: int,
    dim: int,
    seed: int,
    bounds: Dict[Split, Tuple[float, float]],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(total, seq_len, dim, generator=generator)
    ticks = torch.linspace(0, 1, seq_len)
    sinusoid = torch.sin(2 * math.pi * ticks).unsqueeze(0).unsqueeze(-1)
    ramp = torch.linspace(-1, 1, seq_len).unsqueeze(0).unsqueeze(-1)
    base[:, :, 0:1] += 0.75 * sinusoid
    base[:, :, 1:2] += 0.5 * torch.cos(4 * math.pi * ticks).unsqueeze(0).unsqueeze(-1)
    base[:, :, 2:3] += 0.25 * ramp
    start_f, end_f = bounds.get(split, (0.0, 1.0))
    start = int(start_f * total)
    end = int(end_f * total)
    subset = base[start:end].clone().contiguous()
    return subset, None


def _load_or_create_hdf5(
    path: Path,
    split: Split,
    synth_cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if h5py is None:
        raise RuntimeError(
            "h5py is required to read datasets from disk. Install h5py or omit the path "
            "to fall back to the synthetic micro-step dataset."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _write_demo_hdf5(path, synth_cfg)
    with h5py.File(path, "r") as handle:
        if split in handle:
            group = handle[split]
        else:
            group = handle
        data = torch.from_numpy(group["data"][...]).float()
        labels = torch.from_numpy(group["labels"][...]).float() if "labels" in group else None
    return data, labels


def _write_demo_hdf5(path: Path, synth_cfg: Dict[str, Any]) -> None:
    if h5py is None:
        return
    data, _ = _synthesize_split("train", **synth_cfg)
    val, _ = _synthesize_split("val", **synth_cfg)
    test, _ = _synthesize_split("test", **synth_cfg)
    with h5py.File(path, "w") as handle:
        for name, tensor in ("train", data), ("val", val), ("test", test):
            grp = handle.create_group(name)
            grp.create_dataset("data", data=tensor.numpy(), compression="gzip")

