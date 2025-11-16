# imprint/masks.py

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import torch


def _infer_length(indices: Iterable[int]) -> int:
    maximum = -1
    for idx in indices:
        maximum = max(maximum, int(idx))
    if maximum < 0:
        raise ValueError("Cannot infer mask length from an empty set of indices.")
    return maximum + 1


def _ensure_size(size: Optional[int], inferred: Optional[int]) -> int:
    if size is not None:
        return int(size)
    if inferred is None:
        raise ValueError("size must be provided when it cannot be inferred.")
    return inferred


def indices_zero(indices: Iterable[int], size: Optional[int] = None) -> torch.Tensor:
    """
    Binary mask that keeps every position except the provided indices.

    Args:
        indices: Indices to zero out (set to 0 in the mask).
        size: Optional total length. If omitted, inferred from indices.
    """
    index_list = [int(i) for i in indices]
    length = _ensure_size(size, _infer_length(index_list) if index_list else None)
    mask = torch.ones(length, dtype=torch.float32)
    if index_list:
        idx_tensor = torch.tensor(index_list, dtype=torch.long)
        mask[idx_tensor] = 0.0
    return mask


def keep_every_kth(k: int, size: Optional[int] = None) -> torch.Tensor:
    """
    Binary mask that keeps every k-th element (starting at index 0) and zeros others.
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    if size is None:
        raise ValueError("size must be specified for keep_every_kth.")
    length = int(size)
    mask = torch.zeros(length, dtype=torch.float32)
    mask[::k] = 1.0
    return mask


def band(bandwidth: int, size: Optional[int] = None) -> torch.Tensor:
    """
    Square banded mask (1s along the main diagonal within +/- bandwidth, 0 elsewhere).
    """
    if bandwidth < 0:
        raise ValueError("bandwidth must be >= 0")
    if size is None:
        raise ValueError("size must be provided for band masks.")
    n = int(size)
    rows = torch.arange(n).unsqueeze(0)
    cols = torch.arange(n).unsqueeze(1)
    mask = (torch.abs(rows - cols) <= bandwidth).to(torch.float32)
    return mask


def eye(n_out: int, n_in: int) -> torch.Tensor:
    """
    Rectangular identity mask with shape [n_out, n_in].
    """
    if n_out <= 0 or n_in <= 0:
        raise ValueError("n_out and n_in must be > 0.")
    mask = torch.zeros(n_out, n_in, dtype=torch.float32)
    diag = torch.arange(min(n_out, n_in))
    mask[diag, diag] = 1.0
    return mask


def random_sparsity(p: float, shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
    """
    Random binary sparsity mask with keep probability p.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("Probability p must be in [0, 1].")
    if shape is None:
        raise ValueError("shape must be specified for random_sparsity.")
    return torch.bernoulli(torch.full(shape, float(p)))


def complement_of(mask_or_tensor: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
    """
    Complement (1 - mask) of a supplied tensor or sequence of tensors.

    Args:
        mask_or_tensor: Tensor or sequence of tensors representing masks.
                         When a sequence is provided, they are broadcast-summed first.
    """
    if isinstance(mask_or_tensor, torch.Tensor):
        mask = mask_or_tensor
    elif isinstance(mask_or_tensor, Sequence):
        if not mask_or_tensor:
            raise ValueError("Cannot build complement from an empty sequence.")
        tensors = [torch.as_tensor(t, dtype=torch.float32) for t in mask_or_tensor]
        mask = torch.stack(tensors, dim=0).sum(dim=0).clamp(max=1.0)
    else:
        raise TypeError("complement_of expects a tensor or a sequence of tensors.")
    return (1.0 - torch.as_tensor(mask, dtype=torch.float32)).clamp(min=0.0, max=1.0)
