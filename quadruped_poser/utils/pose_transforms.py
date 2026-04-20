"""High-level pose transformation utilities.

Provides conversions between axis-angle, rotation matrix, quaternion, and
Euler-angle representations, plus pose-level helpers for stripping and
restoring global orientation components.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .rot_conversions import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis


# ---------------------------------------------------------------------------
# Kinematic helpers
# ---------------------------------------------------------------------------

def local2global_pose(local_pose: torch.Tensor, kintree) -> torch.Tensor:
    """Convert local (relative) joint rotation matrices to global (absolute) ones.

    Traverses the kinematic tree in parent-first order, accumulating rotations
    by left-multiplying each joint's local rotation with its parent's global
    rotation.  The root joint (index 0) is assumed to have no parent, so its
    global rotation equals its local rotation.

    Args:
        local_pose: Batch of local rotation matrices with shape ``(B, J*9)`` or
            ``(B, J, 3, 3)``, where ``B`` is the batch size and ``J`` is the
            number of joints.  The tensor is reshaped internally to
            ``(B, J, 3, 3)``.
        kintree: Sequence of length ``J`` where ``kintree[j]`` is the index of
            joint ``j``'s parent.  A value ``< 0`` indicates a root joint
            (no parent).

    Returns:
        Global rotation matrices with shape ``(B, J, 3, 3)``.  Each matrix
        ``global_pose[:, j]`` is the rotation that maps joint-``j``-local
        coordinates to the world frame.
    """
    bs = local_pose.shape[0]
    local_pose = local_pose.view(bs, -1, 3, 3)
    global_pose = local_pose.clone()
    for jId in range(len(kintree)):
        parent = kintree[jId]
        if parent >= 0:
            global_pose[:, jId] = torch.matmul(global_pose[:, parent], global_pose[:, jId])
    return global_pose


def batch_rigid_transform(rot_mats: torch.Tensor, joints: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """Apply a batch of rigid transforms to joint locations.

    Builds a kinematic chain of 4×4 homogeneous transform matrices by
    composing the rotation at each joint with the relative offset from its
    parent joint.  The chain is accumulated in topological (parent-first)
    order so each joint's world transform equals the product of all ancestor
    transforms.

    Args:
        rot_mats: Per-joint rotation matrices with shape ``(B, N, 3, 3)``,
            where ``B`` is the batch size and ``N`` is the number of joints.
        joints: Rest-pose joint positions with shape ``(B, N, 3)``.
        parents: Kinematic parent index for each joint, shape ``(N,)``.
            ``parents[0]`` is typically ``-1`` (root) and is assumed to be
            the first element of ``chain``.

    Returns:
        Posed joint positions with shape ``(B, N, 3)``.
    """
    from .rot_conversions import angle_axis_to_rotation_matrix  # local import to avoid circular

    def tmat(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Build a (B, 4, 4) homogeneous transform from rotation R and translation t."""
        bs = R.shape[0]
        T = torch.zeros(bs, 1, 4, 4, device=R.device, dtype=R.dtype)
        T[:, 0, :3, :3] = R
        T[:, 0, :3,  3] = t.squeeze(-1)
        T[:, 0,  3,  3] = 1.0
        return T

    joints = joints.unsqueeze(-1)
    rel = joints.clone()
    rel[:, 1:] -= joints[:, parents[1:]]

    chain = [tmat(rot_mats[:, 0], rel[:, 0])[:, 0]]
    for i in range(1, parents.shape[0]):
        chain.append(torch.matmul(chain[parents[i]], tmat(rot_mats[:, i], rel[:, i])[:, 0]))

    return torch.stack(chain, dim=1)[:, :, :3, 3]


# ---------------------------------------------------------------------------
# Axis-angle <-> rotation matrix
# ---------------------------------------------------------------------------

def matrot2aa(rot_mat: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to axis-angle vectors.

    The conversion proceeds in two steps:

    1. **Homogeneous padding** — each ``(3, 3)`` rotation matrix is padded with
       a column of zeros to produce a ``(3, 4)`` matrix that
       ``rotation_matrix_to_angle_axis`` expects as a homogeneous transform
       (the missing last row ``[0, 0, 0, 1]`` is implicitly handled by the
       underlying implementation).
    2. **Matrix → axis-angle** — ``rotation_matrix_to_angle_axis`` extracts
       the axis-angle vector whose direction is the rotation axis and whose
       magnitude is the rotation angle in radians.

    Args:
        rot_mat: Batch of ``(3, 3)`` rotation matrices with shape ``(N, 3, 3)``.

    Returns:
        Batch of axis-angle vectors with shape ``(N, 3)``.  The vector
        ``v = matrot2aa(R)`` satisfies ``R = aa2matrot(v)`` up to numerical
        precision.
    """
    homog = F.pad(rot_mat, [0, 1])
    return rotation_matrix_to_angle_axis(homog)


def aa2matrot(pose: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors to rotation matrices.

    The conversion proceeds in two steps:

    1. **Axis-angle → 4×4 homogeneous matrix** — ``angle_axis_to_rotation_matrix``
       produces a ``(N, 4, 4)`` homogeneous rotation matrix using the
       Rodrigues rotation formula:

       .. code-block:: text

           R = I + sin(θ) * K + (1 - cos(θ)) * K²

       where ``θ = ‖v‖`` is the rotation angle, ``K`` is the skew-symmetric
       cross-product matrix of the unit axis ``v / θ``, and ``I`` is the
       identity matrix.

    2. **Slice to 3×3** — the upper-left ``(3, 3)`` block of each homogeneous
       matrix is extracted and returned as a contiguous tensor.

    Args:
        pose: Batch of axis-angle vectors with shape ``(N, 3)``.  The direction
            encodes the rotation axis and the magnitude encodes the rotation
            angle in radians.

    Returns:
        Batch of ``(3, 3)`` rotation matrices with shape ``(N, 3, 3)``.
    """
    return angle_axis_to_rotation_matrix(pose)[:, :3, :3].contiguous()


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Return sqrt(x) for positive elements and 0 elsewhere, grad-safe."""
    ret = torch.zeros_like(x)
    pos = x > 0
    if torch.is_grad_enabled():
        ret[pos] = torch.sqrt(x[pos])
    else:
        ret = torch.where(pos, torch.sqrt(x), ret)
    return ret


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors to unit quaternions.

    Each axis-angle vector ``v`` encodes a rotation of angle ``θ = ‖v‖``
    around axis ``v / θ``.  The corresponding unit quaternion is:

    .. code-block:: text

        q = [cos(θ/2),  sin(θ/2) * (v / θ)]
          = [cos(θ/2),  sinc(θ/2 / π) * v * 0.5]

    where ``sinc(x) = sin(πx) / (πx)`` (the normalised sinc) is used to
    handle the ``θ → 0`` limit without a branch.  The resulting quaternion is
    in ``(w, x, y, z)`` order.

    Args:
        axis_angle: Axis-angle vectors with shape ``(..., 3)``.

    Returns:
        Unit quaternions with shape ``(..., 4)`` in ``(w, x, y, z)`` order.
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    scale = 0.5 * torch.sinc(angles * 0.5 / torch.pi)
    return torch.cat([torch.cos(angles * 0.5), axis_angle * scale], dim=-1)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions to ``(3, 3)`` rotation matrices.

    Uses the standard formula for the rotation matrix corresponding to a unit
    quaternion ``q = (r, i, j, k)`` (``w, x, y, z`` order):

    .. code-block:: text

        s = 2 / ‖q‖²
        R = | 1 - s(j²+k²)   s(ij - kr)    s(ik + jr) |
            | s(ij + kr)     1 - s(i²+k²)  s(jk - ir) |
            | s(ik - jr)     s(jk + ir)    1 - s(i²+j²)|

    Args:
        q: Unit quaternions with shape ``(..., 4)`` in ``(w, x, y, z)`` order.

    Returns:
        Rotation matrices with shape ``(..., 3, 3)``.
    """
    r, i, j, k = torch.unbind(q, -1)
    s = 2.0 / (q * q).sum(-1)
    o = torch.stack([
        1 - s*(j*j + k*k),  s*(i*j - k*r),      s*(i*k + j*r),
        s*(i*j + k*r),      1 - s*(i*i + k*k),  s*(j*k - i*r),
        s*(i*k - j*r),      s*(j*k + i*r),      1 - s*(i*i + j*j),
    ], -1)
    return o.reshape(q.shape[:-1] + (3, 3))


def standardize_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Flip quaternions so that the scalar (w) component is non-negative.

    Every rotation can be represented by two antipodal quaternions ``q`` and
    ``-q``.  This function selects the canonical representative with
    ``w >= 0``, which is required by some downstream conversions (e.g.
    ``matrix_to_quaternion``).

    Args:
        q: Quaternions with shape ``(..., 4)`` in ``(w, x, y, z)`` order.

    Returns:
        Standardised quaternions with shape ``(..., 4)`` where ``w >= 0``.
    """
    return torch.where(q[..., 0:1] < 0, -q, q)


def matrix_to_quaternion(mat: torch.Tensor) -> torch.Tensor:
    """Convert ``(3, 3)`` rotation matrices to unit quaternions.

    Uses the numerically stable Shepperd method: the component of the
    quaternion with the largest absolute value is identified first (via
    ``_sqrt_positive_part``), and the remaining components are derived
    from that dominant component to avoid division by near-zero values.

    The output quaternion is standardised so that ``w >= 0`` via
    :func:`standardize_quaternion`.

    Args:
        mat: Rotation matrices with shape ``(..., 3, 3)``.

    Returns:
        Unit quaternions with shape ``(..., 4)`` in ``(w, x, y, z)`` order.

    Raises:
        ValueError: If the last two dimensions of ``mat`` are not ``(3, 3)``.
    """
    if mat.size(-1) != 3 or mat.size(-2) != 3:
        raise ValueError(f"Expected (..., 3, 3), got {mat.shape}")
    batch = mat.shape[:-2]
    m = mat.reshape(batch + (9,))
    m00,m01,m02,m10,m11,m12,m20,m21,m22 = torch.unbind(m, -1)
    q_abs = _sqrt_positive_part(torch.stack([
        1+m00+m11+m22, 1+m00-m11-m22, 1-m00+m11-m22, 1-m00-m11+m22,
    ], -1))
    by_rijk = torch.stack([
        torch.stack([q_abs[...,0]**2, m21-m12, m02-m20, m10-m01], -1),
        torch.stack([m21-m12, q_abs[...,1]**2, m10+m01, m02+m20], -1),
        torch.stack([m02-m20, m10+m01, q_abs[...,2]**2, m12+m21], -1),
        torch.stack([m10-m01, m20+m02, m21+m12, q_abs[...,3]**2], -1),
    ], -2)
    flr = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    candidates = by_rijk / (2.0 * q_abs[..., None].max(flr))
    out = candidates[F.one_hot(q_abs.argmax(-1), 4) > 0.5, :].reshape(batch + (4,))
    return standardize_quaternion(out)


def quaternion_to_axis_angle(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions to axis-angle vectors.

    Inverts :func:`axis_angle_to_quaternion` using:

    .. code-block:: text

        θ/2 = atan2(‖q_xyz‖, q_w)
        v   = q_xyz / sinc(θ/2 / π)

    where ``sinc(x) = sin(πx) / (πx)`` (normalised sinc) handles the
    ``θ → 0`` limit smoothly.  The returned vector has direction equal to the
    rotation axis and magnitude equal to the rotation angle ``θ`` in radians.

    Args:
        q: Unit quaternions with shape ``(..., 4)`` in ``(w, x, y, z)`` order.

    Returns:
        Axis-angle vectors with shape ``(..., 3)``.
    """
    norms = torch.norm(q[..., 1:], p=2, dim=-1, keepdim=True)
    half  = torch.atan2(norms, q[..., :1])
    scale = 0.5 * torch.sinc(half / torch.pi)
    return q[..., 1:] / scale


def axis_angle_to_matrix(axis_angle: torch.Tensor, fast: bool = False) -> torch.Tensor:
    """Convert axis-angle vectors to ``(3, 3)`` rotation matrices.

    Two implementations are provided:

    * **Default (``fast=False``)** — chains :func:`axis_angle_to_quaternion`
      and :func:`quaternion_to_matrix`.  This path is numerically robust and
      supports automatic differentiation.

    * **Fast (``fast=True``)** — applies the Rodrigues rotation formula
      directly via the skew-symmetric cross-product matrix ``K``:

      .. code-block:: text

          R = I + sinc(θ/π) * K + (1 - cos(θ)) / θ² * K²

      where ``θ = ‖v‖``.  Slightly faster but may be less numerically stable
      for very small angles.

    Args:
        axis_angle: Axis-angle vectors with shape ``(..., 3)``.
        fast: If ``True``, use the direct Rodrigues path.  Defaults to
            ``False``.

    Returns:
        Rotation matrices with shape ``(..., 3, 3)``.
    """
    if not fast:
        return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))
    shape = axis_angle.shape
    device, dtype = axis_angle.device, axis_angle.dtype
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True).unsqueeze(-1)
    rx, ry, rz = axis_angle[..., 0], axis_angle[..., 1], axis_angle[..., 2]
    zeros = torch.zeros(shape[:-1], dtype=dtype, device=device)
    K = torch.stack([zeros,-rz,ry, rz,zeros,-rx, -ry,rx,zeros], -1).view(shape+(3,))
    K2 = K @ K
    I = torch.eye(3, dtype=dtype, device=device)
    a2 = angles*angles
    a2 = torch.where(a2 == 0, torch.ones_like(a2), a2)
    return I.expand(K.shape) + torch.sinc(angles/torch.pi)*K + ((1-torch.cos(angles))/a2)*K2


# ---------------------------------------------------------------------------
# Euler angle utilities
# ---------------------------------------------------------------------------

def _index_from_letter(letter: str) -> int:
    """Return the 0-based axis index for a single letter 'X', 'Y', or 'Z'."""
    return {'X': 0, 'Y': 1, 'Z': 2}[letter]


def _angle_from_tan(axis: str, other_axis: str, data: torch.Tensor,
                    horizontal: bool, tait_bryan: bool) -> torch.Tensor:
    """Compute one Euler angle from the appropriate matrix row/column via atan2."""
    i1, i2 = {'X': (2,1), 'Y': (0,2), 'Z': (1,0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ('XY', 'YZ', 'ZX')
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """Build elemental rotation matrices for a single named axis and angle tensor."""
    c, s = torch.cos(angle), torch.sin(angle)
    one, zero = torch.ones_like(angle), torch.zeros_like(angle)
    flat = {'X': (one,zero,zero,zero,c,-s,zero,s,c),
            'Y': (c,zero,s,zero,one,zero,-s,zero,c),
            'Z': (c,-s,zero,s,c,zero,zero,zero,one)}[axis]
    return torch.stack(flat, -1).reshape(angle.shape + (3, 3))


def matrix_to_euler_angles(mat: torch.Tensor, convention: str) -> torch.Tensor:
    """Decompose rotation matrices into Euler angles for a given convention.

    Supports any valid three-axis Tait-Bryan or proper Euler convention (e.g.
    ``'XYZ'``, ``'ZYX'``, ``'ZXZ'``).  The central angle is extracted using
    ``asin`` (Tait-Bryan) or ``acos`` (proper Euler), then the first and last
    angles are recovered via ``atan2`` on the appropriate matrix elements.

    Note: The decomposition is singular (gimbal lock) when the central angle
    approaches ±π/2 (Tait-Bryan) or 0/π (proper Euler).

    Args:
        mat: Rotation matrices with shape ``(..., 3, 3)``.
        convention: Three-character string such as ``'XYZ'`` specifying the
            rotation order.  The middle character must differ from both the
            first and last characters.

    Returns:
        Euler angles with shape ``(..., 3)`` in radians, ordered according to
        ``convention``.

    Raises:
        ValueError: If ``convention`` is not a valid three-axis convention.
    """
    if len(convention) != 3 or convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention: {convention}")
    i0, i2 = _index_from_letter(convention[0]), _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    central = (torch.asin(mat[..., i0, i2] * (-1.0 if i0-i2 in (-1, 2) else 1.0))
               if tait_bryan else torch.acos(mat[..., i0, i0]))
    return torch.stack([
        _angle_from_tan(convention[0], convention[1], mat[..., i2],    False, tait_bryan),
        central,
        _angle_from_tan(convention[2], convention[1], mat[..., i0, :], True,  tait_bryan),
    ], -1)


def euler_angles_to_matrix(euler: torch.Tensor, convention: str) -> torch.Tensor:
    """Convert Euler angles to rotation matrices.

    Builds three elemental rotation matrices (one per axis) and multiplies
    them left-to-right in the order given by ``convention``:

    .. code-block:: text

        R = R_convention[0] @ R_convention[1] @ R_convention[2]

    where each elemental matrix is built by :func:`_axis_angle_rotation`.

    Args:
        euler: Euler angles with shape ``(..., 3)`` in radians.  The last
            dimension must have exactly 3 elements.
        convention: Three-character string specifying the rotation order,
            e.g. ``'XYZ'``.  Must have exactly 3 characters.

    Returns:
        Rotation matrices with shape ``(..., 3, 3)``.

    Raises:
        ValueError: If ``euler`` shape or ``convention`` length is invalid.
    """
    if euler.dim() == 0 or euler.shape[-1] != 3 or len(convention) != 3:
        raise ValueError(f"Invalid euler angles shape {euler.shape} or convention {convention}")
    mats = [_axis_angle_rotation(c, e) for c, e in zip(convention, torch.unbind(euler, -1))]
    return torch.matmul(torch.matmul(mats[0], mats[1]), mats[2])


def matrix_to_axis_angle(mat: torch.Tensor, fast: bool = False) -> torch.Tensor:
    """Convert ``(3, 3)`` rotation matrices to axis-angle vectors.

    Two implementations are provided:

    * **Default (``fast=False``)** — chains :func:`matrix_to_quaternion` and
      :func:`quaternion_to_axis_angle`.  Numerically robust across all
      rotation angles.

    * **Fast (``fast=True``)** — uses the analytic formula:

      .. code-block:: text

          ω = [R₃₂ - R₂₃, R₁₃ - R₃₁, R₂₁ - R₁₂]
          θ = atan2(‖ω‖, tr(R) - 1)
          v = θ / sinc(θ/π) * (ω / ‖ω‖)    (for θ not near π)

      A special branch handles the near-π case where ``ω → 0`` using the
      diagonal entries of ``R``.

    Args:
        mat: Rotation matrices with shape ``(..., 3, 3)``.
        fast: If ``True``, use the direct analytic path.  Defaults to
            ``False``.

    Returns:
        Axis-angle vectors with shape ``(..., 3)``.

    Raises:
        ValueError: If the last two dimensions of ``mat`` are not ``(3, 3)``
            and ``fast=True``.
    """
    if not fast:
        return quaternion_to_axis_angle(matrix_to_quaternion(mat))
    if mat.size(-1) != 3 or mat.size(-2) != 3:
        raise ValueError(f"Expected (..., 3, 3), got {mat.shape}")
    omegas = torch.stack([mat[...,2,1]-mat[...,1,2], mat[...,0,2]-mat[...,2,0],
                          mat[...,1,0]-mat[...,0,1]], -1)
    norms  = torch.norm(omegas, p=2, dim=-1, keepdim=True)
    traces = torch.diagonal(mat, dim1=-2, dim2=-1).sum(-1).unsqueeze(-1)
    angles = torch.atan2(norms, traces - 1)
    zeros  = torch.zeros(3, dtype=mat.dtype, device=mat.device)
    omegas = torch.where(torch.isclose(angles, torch.zeros_like(angles)), zeros, omegas)
    near_pi = angles.isclose(angles.new_full((1,), torch.pi)).squeeze(-1)
    out = torch.empty_like(omegas)
    out[~near_pi] = 0.5 * omegas[~near_pi] / torch.sinc(angles[~near_pi] / torch.pi)
    if near_pi.any():
        n = 0.5 * (mat[near_pi][..., 0, :] + torch.eye(1, 3, dtype=mat.dtype, device=mat.device))
        out[near_pi] = angles[near_pi] * n / torch.norm(n)
    return out


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def aa2euler(axis_angle: torch.Tensor, convention: str = 'XYZ') -> torch.Tensor:
    """Convert axis-angle vectors to Euler angles.

    This is a two-step convenience wrapper:

    1. **Axis-angle → rotation matrix** via :func:`axis_angle_to_matrix`.
    2. **Rotation matrix → Euler angles** via :func:`matrix_to_euler_angles`.

    Args:
        axis_angle: Axis-angle vectors with shape ``(..., 3)``.  Direction
            encodes the rotation axis; magnitude encodes the angle in radians.
        convention: Three-character Euler convention string, e.g. ``'XYZ'``
            (default).  Determines the decomposition order used in step 2.

    Returns:
        Euler angles with shape ``(..., 3)`` in radians, ordered according to
        ``convention``.
    """
    return matrix_to_euler_angles(axis_angle_to_matrix(axis_angle), convention)


def euler2aa(euler: torch.Tensor, convention: str = 'XYZ') -> torch.Tensor:
    """Convert Euler angles to axis-angle vectors.

    This is a two-step convenience wrapper:

    1. **Euler angles → rotation matrix** via :func:`euler_angles_to_matrix`.
    2. **Rotation matrix → axis-angle** via :func:`matrix_to_axis_angle`.

    Args:
        euler: Euler angles with shape ``(..., 3)`` in radians, ordered
            according to ``convention``.
        convention: Three-character Euler convention string, e.g. ``'XYZ'``
            (default).  Must match the convention used to produce ``euler``.

    Returns:
        Axis-angle vectors with shape ``(..., 3)``.  Direction encodes the
        rotation axis; magnitude encodes the angle in radians.
    """
    return matrix_to_axis_angle(euler_angles_to_matrix(euler, convention))


# ---------------------------------------------------------------------------
# Pose-level helpers
# ---------------------------------------------------------------------------

def remove_rotation_from_axis(full_pose: torch.Tensor, axis: int,
                               convention: str = 'XYZ') -> torch.Tensor:
    """Zero the global rotation around one axis in a batch of poses.

    When encoding animal poses with a VAE it is desirable to remove the
    "heading" component (typically the vertical Y-axis rotation) from the
    global orientation before passing the pose to the encoder.  This prevents
    the latent space from conflating body-shape information with absolute
    facing direction, and makes the learned distribution invariant to the
    animal's heading in the world.

    The function decomposes the global orientation (the first 3 values of
    each pose, stored as an axis-angle vector) into three Euler angles,
    zeroes the component at ``axis``, and re-encodes the modified angles back
    to an axis-angle vector:

    .. code-block:: text

        global_orient (aa)  -->  aa2euler  -->  euler  -->  euler[axis] = 0
            -->  euler2aa  -->  global_orient_stripped (aa)

    Args:
        full_pose: Batch of axis-angle poses with shape ``(B, pose_dim)``.
            The first 3 values of each row are treated as the global
            orientation in axis-angle form; the remaining values are body
            joint rotations and are left untouched.
        axis: Index of the Euler axis to zero out (``0`` = X, ``1`` = Y,
            ``2`` = Z under the default ``'XYZ'`` convention).
        convention: Euler-angle decomposition convention, e.g. ``'XYZ'``
            (default).  The same convention must be used consistently for
            :func:`remove_rotation_from_axis` and
            :func:`merge_global_orients_along_axis`.

    Returns:
        A new tensor with the same shape as ``full_pose`` where the global
        orientation no longer contains any rotation around ``axis``.  All
        other pose parameters are preserved exactly.
    """
    go = full_pose[:, :3]
    euler = aa2euler(go, convention)
    euler[:, axis] = 0.0
    go_new = euler2aa(euler, convention)
    out = full_pose.clone()
    out[:, :3] = go_new
    return out


def merge_global_orients_along_axis(additional: torch.Tensor, base: torch.Tensor,
                                    axis: int) -> torch.Tensor:
    """Copy one Euler axis component of global orientation from one pose into another.

    After a VAE decodes a pose it typically produces a body configuration that
    lacks heading information (because heading was stripped before encoding via
    :func:`remove_rotation_from_axis`).  This function restores the heading by
    copying the Euler angle at ``axis`` from ``additional`` (which carries the
    original or desired heading) into ``base`` (the decoded body pose), while
    keeping all other Euler components of ``base`` unchanged.

    The "merging" operation:

    .. code-block:: text

        base_euler       = aa2euler(base[:, :3])
        additional_euler = aa2euler(additional[:, :3])
        base_euler[:, axis] = additional_euler[:, axis]   # copy one component
        merged_orient    = euler2aa(base_euler)

    Args:
        additional: Batch of poses with shape ``(B, pose_dim)`` that supplies
            the Euler value at ``axis``.  Only the global orientation
            (``additional[:, :3]``) is read; the rest is ignored.
        base: Batch of poses with shape ``(B, pose_dim)`` whose global
            orientation (``base[:, :3]``) is modified.  All other pose
            parameters are preserved.
        axis: Index of the Euler axis to copy from ``additional`` into
            ``base`` (``0`` = X, ``1`` = Y, ``2`` = Z under the default
            ``'XYZ'`` convention used internally).

    Returns:
        A new tensor with the same shape as ``base`` where ``base[:, :3]``
        has been replaced by the merged global orientation and all remaining
        pose parameters match ``base`` exactly.
    """
    base_euler  = aa2euler(base[:, :3])
    addi_euler  = aa2euler(additional[:, :3])
    base_euler[:, axis] = addi_euler[:, axis]
    out = base.clone()
    out[:, :3] = euler2aa(base_euler)
    return out


# ---------------------------------------------------------------------------
# Numpy helper
# ---------------------------------------------------------------------------

def rotate_points_xyz(mesh_v: np.ndarray, Rxyz) -> np.ndarray:
    """Rotate a batch of point clouds by per-frame XYZ Euler angles (degrees).

    Constructs three elemental rotation matrices ``Rx``, ``Ry``, ``Rz`` for
    each frame and applies the combined rotation ``Rz @ Ry @ Rx`` to every
    vertex in that frame.  Angles are supplied in degrees and converted to
    radians internally.

    Args:
        mesh_v: Vertex positions with shape ``(N, V, 3)`` where ``N`` is the
            number of frames and ``V`` is the number of vertices per frame.
        Rxyz: Rotation angles in degrees.  Can be a length-3 array-like
            (same rotation applied to every frame) or an ``(N, 3)`` array-like
            with per-frame angles.  The three values correspond to rotations
            around the X, Y, and Z axes respectively.

    Returns:
        Rotated vertex array with the same shape ``(N, V, 3)`` as ``mesh_v``.
        Returns an empty array of the same shape if ``mesh_v`` has no frames.
    """
    Rxyz = np.repeat(np.array(Rxyz).reshape(1, 3), len(mesh_v), axis=0)
    rotated = []
    for f in range(mesh_v.shape[0]):
        ax, ay, az = (np.radians(Rxyz[f, i]) for i in range(3))
        Rx = np.array([[1,0,0],[0,np.cos(ax),-np.sin(ax)],[0,np.sin(ax),np.cos(ax)]])
        Ry = np.array([[np.cos(ay),0,np.sin(ay)],[0,1,0],[-np.sin(ay),0,np.cos(ay)]])
        Rz = np.array([[np.cos(az),-np.sin(az),0],[np.sin(az),np.cos(az),0],[0,0,1]])
        rotated.append(Rz @ Ry @ Rx @ mesh_v[f].T)
    return np.array(rotated).transpose(0, 2, 1) if rotated else np.empty_like(mesh_v)


# ---------------------------------------------------------------------------
# Sampling helper (moved from project root)
# ---------------------------------------------------------------------------

def random_rotation_matrices(n: int) -> np.ndarray:
    """Sample uniformly random 3×3 rotation matrices.

    Generates rotation matrices drawn uniformly from SO(3) using
    scipy's Rotation.random, which uses the Haar measure.

    Args:
        n: Number of rotation matrices to generate.

    Returns:
        Array of shape (n, 3, 3) containing valid rotation matrices.

    Example:
        >>> mats = random_rotation_matrices(4)   # (4, 3, 3)
        >>> import numpy as np
        >>> np.allclose(mats @ mats.transpose(0, 2, 1), np.eye(3), atol=1e-6)
        True
    """
    from scipy.spatial.transform import Rotation
    return Rotation.random(n).as_matrix()
