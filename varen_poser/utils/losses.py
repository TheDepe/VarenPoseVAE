"""Rotation-aware loss functions for pose reconstruction."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _gram_schmidt(M: np.ndarray) -> np.ndarray:
    """Orthonormalise two columns of a 3×2 matrix via Gram-Schmidt."""
    a1 = M[:, 0]
    a2 = M[:, 1]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.vstack([b1, b2, b3]).T


def _batch_gram_schmidt(d6: torch.Tensor) -> torch.Tensor:
    """Batch Gram-Schmidt orthonormalisation of a (B, 3, 2) tensor.

    Returns a (B, 3, 3) batch of rotation matrices.
    """
    b = d6.shape[0]
    b1 = F.normalize(d6[:, :, 0], p=2, dim=1)
    a2 = d6[:, :, 1]
    proj = torch.bmm(b1.view(b, 1, -1), a2.view(b, -1, 1)).view(b, 1) * b1
    b2 = F.normalize(a2 - proj, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)


class GeodesicRotationLoss(nn.Module):
    """Geodesic distance loss between batches of rotation matrices.

    Args:
        reduction: ``'mean'`` averages over all elements; any other string
            returns per-element angles.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.eps = 1e-6

    def _pairwise_geodesic(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        n = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))
        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1.0) / 2.0
        cos = torch.clamp(cos, -1.0, 1.0)
        return torch.acos(cos)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        angles = self._pairwise_geodesic(pred, target)
        if self.reduction == 'mean':
            return torch.mean(angles)
        return angles
