"""
Lightweight helpers for loading or synthesizing seq2seq demo datasets.

The goal is to keep demo scripts free of data boilerplate: a single call
returns tensors plus metadata such as feature dimensions and batches/epoch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, Union

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


def load_micro_step_demo_dataset(
    *,
    path: Optional[PathLike] = None,
    split: Split = "train",
    batch_size: int = 16,
) -> SequenceDataset:
    """
    Load the dataset used by demo #4 (micro-stepping) or synthesize one on-demand.

    Args:
        path: Optional path to an HDF5 file following DATASETS.md conventions.
              If omitted, a tiny synthetic dataset is created in-memory.
        split: One of ``{'train', 'val', 'test'}``.
        batch_size: Mini-batch size used by the demo/training loop.
    """
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be 'train', 'val', or 'test'.")

    if path is None:
        data, labels = _synthesize_split(split)
    else:
        dataset_path = Path(path)
        data, labels = _load_or_create_hdf5(dataset_path, split)

    return SequenceDataset(
        data=data,
        labels=labels,
        batch_size=batch_size,
        split=split,
        task_type="self_supervised_seq2seq",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _synthesize_split(split: Split) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    total, seq_len, dim = 320, 160, 64
    generator = torch.Generator().manual_seed(7)
    base = torch.randn(total, seq_len, dim, generator=generator)
    ticks = torch.linspace(0, 1, seq_len)
    sinusoid = torch.sin(2 * math.pi * ticks).unsqueeze(0).unsqueeze(-1)
    ramp = torch.linspace(-1, 1, seq_len).unsqueeze(0).unsqueeze(-1)
    base[:, :, 0:1] += 0.75 * sinusoid
    base[:, :, 1:2] += 0.5 * torch.cos(4 * math.pi * ticks).unsqueeze(0).unsqueeze(-1)
    base[:, :, 2:3] += 0.25 * ramp
    bounds = {"train": (0.0, 0.7), "val": (0.7, 0.85), "test": (0.85, 1.0)}
    start_f, end_f = bounds[split]
    start = int(start_f * total)
    end = int(end_f * total)
    subset = base[start:end].clone().contiguous()
    return subset, None


def _load_or_create_hdf5(path: Path, split: Split) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if h5py is None:
        raise RuntimeError(
            "h5py is required to read datasets from disk. Install h5py or omit the path "
            "to fall back to the synthetic micro-step dataset."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _write_demo_hdf5(path)
    with h5py.File(path, "r") as handle:
        if split in handle:
            group = handle[split]
        else:
            group = handle
        data = torch.from_numpy(group["data"][...]).float()
        labels = torch.from_numpy(group["labels"][...]).float() if "labels" in group else None
    return data, labels


def _write_demo_hdf5(path: Path) -> None:
    if h5py is None:
        return
    data, _ = _synthesize_split("train")
    val, _ = _synthesize_split("val")
    test, _ = _synthesize_split("test")
    with h5py.File(path, "w") as handle:
        for name, tensor in ("train", data), ("val", val), ("test", test):
            grp = handle.create_group(name)
            grp.create_dataset("data", data=tensor.numpy(), compression="gzip")

