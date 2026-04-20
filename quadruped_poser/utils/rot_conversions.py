"""Low-level axis-angle ↔ rotation-matrix conversions.

Uses the Rodrigues rotation formula for numerically stable conversion.
A Taylor approximation is applied near the origin (small-angle case) to
avoid division by zero.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def angle_axis_to_rotation_matrix(angle_axis: torch.Tensor) -> torch.Tensor:
    """Convert a batch of axis-angle vectors to 4×4 rotation matrices.

    Each axis-angle vector encodes a rotation whose direction is the unit
    axis of rotation and whose magnitude is the rotation angle in radians.
    The upper-left 3×3 block of the output is the SO(3) rotation matrix
    computed via the Rodrigues formula; the bottom row and right column
    form the homogeneous padding ``[0, 0, 0, 1]``.

    For vectors whose squared norm falls below ``eps = 1e-6`` a first-order
    Taylor expansion is used instead of the full Rodrigues formula to avoid
    numerical instability near the identity.

    Args:
        angle_axis: Axis-angle vectors of shape ``(N, 3)``.  The magnitude
            of each row encodes the rotation angle in radians.

    Returns:
        Homogeneous rotation matrices of shape ``(N, 4, 4)``.  The
        translation column is zero and the bottom row is
        ``[0, 0, 0, 1]``.

    Note:
        Rodrigues formula:
        ``R = I cos θ + (1 − cos θ) w wᵀ + sin θ [w]×``
        where ``w = v / ‖v‖`` and ``θ = ‖v‖``.
    """
    def _rodrigues(aa: torch.Tensor, theta2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Apply the full Rodrigues rotation formula element-wise."""
        theta = torch.sqrt(theta2)
        wxyz = aa / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        c, s = torch.cos(theta), torch.sin(theta)
        k = 1.0
        r00 = c + wx*wx*(k-c);      r01 = wx*wy*(k-c) - wz*s;  r02 = wy*s  + wx*wz*(k-c)
        r10 = wz*s + wx*wy*(k-c);   r11 = c + wy*wy*(k-c);     r12 = -wx*s + wy*wz*(k-c)
        r20 = -wy*s + wx*wz*(k-c);  r21 = wx*s + wy*wz*(k-c);  r22 = c + wz*wz*(k-c)
        return torch.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1).view(-1, 3, 3)

    def _taylor(aa: torch.Tensor) -> torch.Tensor:
        """Apply the first-order Taylor approximation for small angles."""
        rx, ry, rz = torch.chunk(aa, 3, dim=1)
        one = torch.ones_like(rx)
        return torch.cat([one, -rz, ry, rz, one, -rx, -ry, rx, one], dim=1).view(-1, 3, 3)

    _aa   = angle_axis.unsqueeze(1)
    theta2 = torch.squeeze(torch.matmul(_aa, _aa.transpose(1, 2)), dim=1)

    mat_rodrigues = _rodrigues(angle_axis, theta2)
    mat_taylor    = _taylor(angle_axis)

    eps  = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).type_as(theta2)

    bs = angle_axis.shape[0]
    out = torch.eye(4, device=angle_axis.device, dtype=angle_axis.dtype).view(1, 4, 4).expand(bs, -1, -1).clone()
    out[:, :3, :3] = mask * mat_rodrigues + (1.0 - mask) * mat_taylor
    return out


def rotation_matrix_to_angle_axis(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """Convert a batch of 3×4 rotation matrices to axis-angle vectors.

    This is a two-step conversion: the rotation matrix is first mapped to
    a unit quaternion via :func:`rotation_matrix_to_quaternion`, and the
    quaternion is then converted to an axis-angle vector via
    :func:`quaternion_to_angle_axis`.

    Args:
        rotation_matrix: Rotation matrices of shape ``(N, 3, 4)``.  The
            fourth column (translation) is ignored.

    Returns:
        Axis-angle vectors of shape ``(N, 3)``.  The direction of each
        vector is the rotation axis and its magnitude is the rotation
        angle in radians.
    """
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)


def rotation_matrix_to_quaternion(rotation_matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert a batch of 3×4 rotation matrices to quaternions (w, x, y, z).

    The algorithm selects one of four numerically stable branches depending
    on which diagonal entry of the rotation matrix is largest, following
    Shepperd's method.  Each branch produces a quaternion proportional to
    the correct result; the quaternion is then normalised and scaled to
    unit length.

    Args:
        rotation_matrix: Rotation matrices of shape ``(N, 3, 4)``.  The
            fourth column is ignored.
        eps: Small constant used for the diagonal comparison that selects
            the stable branch.  Defaults to ``1e-6``.

    Returns:
        Unit quaternions of shape ``(N, 4)`` in ``(w, x, y, z)`` order.

    Raises:
        TypeError: If ``rotation_matrix`` is not a ``torch.Tensor``.
        ValueError: If ``rotation_matrix`` does not have shape ``(N, 3, 4)``.

    Note:
        Shepperd's method guarantees that the selected branch always divides
        by a quantity of magnitude ``≥ 1``, avoiding catastrophic
        cancellation.
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError(f"Expected torch.Tensor, got {type(rotation_matrix)}")
    if rotation_matrix.ndim > 3 or rotation_matrix.shape[-2:] != (3, 4):
        raise ValueError(f"Expected shape (N, 3, 4), got {rotation_matrix.shape}")

    rmat_t = rotation_matrix.transpose(1, 2)

    mask_d2    = rmat_t[:, 2, 2] < eps
    mask_d0_d1 = rmat_t[:, 0, 0] >  rmat_t[:, 1, 1]
    mask_d0nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2]-rmat_t[:, 2, 1], t0,
                      rmat_t[:, 0, 1]+rmat_t[:, 1, 0], rmat_t[:, 2, 0]+rmat_t[:, 0, 2]], -1)

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0]-rmat_t[:, 0, 2], rmat_t[:, 0, 1]+rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2]+rmat_t[:, 2, 1]], -1)

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1]-rmat_t[:, 1, 0], rmat_t[:, 2, 0]+rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2]+rmat_t[:, 2, 1], t2], -1)

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2]-rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0]-rmat_t[:, 0, 2], rmat_t[:, 0, 1]-rmat_t[:, 1, 0]], -1)

    mc0 = (mask_d2  *        mask_d0_d1 ).view(-1, 1).type_as(q0)
    mc1 = (mask_d2  * ~mask_d0_d1       ).view(-1, 1).type_as(q1)
    mc2 = (~mask_d2 *        mask_d0nd1 ).view(-1, 1).type_as(q2)
    mc3 = (~mask_d2 * ~mask_d0nd1       ).view(-1, 1).type_as(q3)

    q = q0*mc0 + q1*mc1 + q2*mc2 + q3*mc3
    t_rep = t0.repeat(4, 1).t()*mc0 + t1.repeat(4, 1).t()*mc1 + \
            t2.repeat(4, 1).t()*mc2 + t3.repeat(4, 1).t()*mc3
    q /= torch.sqrt(t_rep)
    q *= 0.5
    return q


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternions (w, x, y, z) to axis-angle vectors.

    Given a unit quaternion ``q = (cos(θ/2), sin(θ/2) * w)`` the
    corresponding axis-angle vector is ``θ * w``, where ``w`` is the unit
    rotation axis and ``θ ∈ [0, π]`` is the rotation angle.  A numerically
    stable ``atan2``-based formula is used to recover ``θ``, and a
    degenerate branch handles the case ``sin(θ/2) ≈ 0`` (near-identity
    rotation) by replacing the division with the limit value ``2``.

    Args:
        quaternion: Unit quaternions of arbitrary batch shape ``(*, 4)``
            in ``(w, x, y, z)`` order.

    Returns:
        Axis-angle vectors of shape ``(*, 3)``.  The magnitude of each
        vector equals the rotation angle in radians.

    Raises:
        ValueError: If ``quaternion`` is not a ``torch.Tensor`` with last
            dimension equal to 4.

    Note:
        The scaling factor is
        ``k = 2 θ / sin(θ/2)``
        which approaches ``2`` as ``θ → 0``, ensuring a smooth and
        differentiable map through the identity rotation.
    """
    if not torch.is_tensor(quaternion) or quaternion.shape[-1] != 4:
        raise ValueError(f"Expected Tensor of shape (*, 4), got {getattr(quaternion, 'shape', type(quaternion))}")

    q1, q2, q3 = quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    sin2  = q1*q1 + q2*q2 + q3*q3
    sin_t = torch.sqrt(sin2)
    cos_t = quaternion[..., 0]

    two_theta = 2.0 * torch.where(
        cos_t < 0.0,
        torch.atan2(-sin_t, -cos_t),
        torch.atan2( sin_t,  cos_t),
    )
    k = torch.where(sin2 > 0.0, two_theta / sin_t, 2.0 * torch.ones_like(sin_t))

    aa = torch.zeros_like(quaternion)[..., :3]
    aa[..., 0] = q1 * k
    aa[..., 1] = q2 * k
    aa[..., 2] = q3 * k
    return aa
