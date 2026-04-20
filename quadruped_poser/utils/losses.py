"""Rotation-aware loss functions for pose reconstruction."""

from __future__ import annotations

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

    The geodesic distance between two rotation matrices ``R₁`` and ``R₂``
    is the angle of the relative rotation ``R₁ Rᵀ₂``, i.e. the shortest
    arc on SO(3) connecting the two orientations.  This metric is more
    appropriate than element-wise MSE for rotation matrices because it
    respects the curved geometry of the rotation group.

    Args:
        reduction: Specifies the reduction applied to per-element angles.
            ``'mean'`` returns the scalar mean over the batch; any other
            string returns the raw per-element angle tensor.

    Example:
        >>> loss_fn = GeodesicRotationLoss(reduction='mean')
        >>> R = torch.eye(3).unsqueeze(0)          # (1, 3, 3)
        >>> loss = loss_fn(R, R)                    # should be 0.0
    """

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction
        self.eps = 1e-6

    def _pairwise_geodesic(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        """Compute per-pair geodesic angles between two batches of rotation matrices."""
        n = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))
        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1.0) / 2.0
        cos = torch.clamp(cos, -1.0, 1.0)
        return torch.acos(cos)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the geodesic rotation loss between predicted and target matrices.

        Evaluates the geodesic distance for each pair ``(pred[i], target[i])``
        and optionally reduces over the batch dimension.

        Args:
            pred: Predicted rotation matrices of shape ``(N, 3, 3)``.
            target: Ground-truth rotation matrices of shape ``(N, 3, 3)``.

        Returns:
            A scalar mean geodesic angle (in radians) when
            ``self.reduction == 'mean'``, otherwise a tensor of shape
            ``(N,)`` containing the per-element geodesic angles.

        Note:
            The geodesic angle is recovered from the relative rotation as
            ``θ = arccos((tr(R₁ Rᵀ₂) − 1) / 2)``, which is the rotation
            angle of the relative rotation matrix ``R₁ Rᵀ₂ ∈ SO(3)``.
        """
        angles = self._pairwise_geodesic(pred, target)
        if self.reduction == 'mean':
            return torch.mean(angles)
        return angles
