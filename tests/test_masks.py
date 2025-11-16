import torch

from imprint import masks


def test_indices_and_keep_masks():
    mask = masks.indices_zero([0, 2], size=5)
    assert torch.equal(mask, torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0]))

    keep = masks.keep_every_kth(2, size=6)
    assert torch.equal(keep, torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]))


def test_band_and_eye_masks():
    eye = masks.eye(3, 4)
    assert eye.shape == (3, 4)
    assert torch.allclose(torch.diag(eye), torch.ones(3))

    band = masks.band(1, size=4)
    assert band.shape == (4, 4)
    assert torch.sum(torch.abs(torch.diag(band, 2))) == 0


def test_random_and_complement():
    rand_mask = masks.random_sparsity(0.5, shape=(4, 4))
    assert rand_mask.shape == (4, 4)
    assert rand_mask.max() <= 1 and rand_mask.min() >= 0

    base = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    comp = masks.complement_of(base)
    assert torch.equal(comp, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
