# imprint/masks.py

from __future__ import annotations
from typing import Iterable, Optional, Tuple
import torch


def indices_zero(indices: Iterable[int], size: Optional[int] = None) -> torch.Tensor:
    """
    Mask that zeros out the given indices (1 for keep, 0 for zero).
    """
    ...


def keep_every_kth(k: int, size: Optional[int] = None) -> torch.Tensor:
    """
    Mask that keeps every k-th index and zeros others.
    """
    ...


def band(bandwidth: int, size: Optional[int] = None) -> torch.Tensor:
    """
    Banded mask for square weight matrices.
    """
    ...


def eye(n_out: int, n_in: int) -> torch.Tensor:
    """
    Identity-like mask for rectangular matrices.
    """
    ...


def random_sparsity(p: float, shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
    """
    Random binary sparsity mask with keep probability p.
    """
    ...


def complement_of(edge_or_name) -> torch.Tensor:
    """
    Placeholder API to build a mask as the complement of another edge's mask.
    Implementation will need access to the graph/edge registry.
    """
    ...
