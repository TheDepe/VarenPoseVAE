"""High-level pose transformation utilities.

Provides conversions between axis-angle, rotation matrix, quaternion, and
Euler-angle representations, plus pose-level helpers for stripping and
restoring global orientation components.
"""

import numpy as np
import torch
from torch.nn import functional as F

from .rot_conversions import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis


# ---------------------------------------------------------------------------
# Kinematic helpers
# ---------------------------------------------------------------------------

def local2global_pose(local_pose: torch.Tensor, kintree) -> torch.Tensor:
    bs = local_pose.shape[0]
    local_pose = local_pose.view(bs, -1, 3, 3)
    global_pose = local_pose.clone()
    for jId in range(len(kintree)):
        parent = kintree[jId]
        if parent >= 0:
            global_pose[:, jId] = torch.matmul(global_pose[:, parent], global_pose[:, jId])
    return global_pose


def batch_rigid_transform(rot_mats: torch.Tensor, joints: torch.Tensor, parents: torch.Tensor):
    """Apply a batch of rigid transforms to joint locations.

    Args:
        rot_mats: (B, N, 3, 3) rotation matrices.
        joints:   (B, N, 3) joint positions.
        parents:  (N,) kinematic parent indices.

    Returns:
        (B, N, 3) posed joint positions.
    """
    from .rot_conversions import angle_axis_to_rotation_matrix  # local import to avoid circular

    def tmat(R, t):
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
# Axis-angle ↔ rotation matrix
# ---------------------------------------------------------------------------

def matrot2aa(rot_mat: torch.Tensor) -> torch.Tensor:
    """Convert (N, 3, 3) rotation matrices to (N, 3) axis-angle vectors."""
    homog = F.pad(rot_mat, [0, 1])
    return rotation_matrix_to_angle_axis(homog)


def aa2matrot(pose: torch.Tensor) -> torch.Tensor:
    """Convert (N, 3) axis-angle vectors to (N, 3, 3) rotation matrices."""
    return angle_axis_to_rotation_matrix(pose)[:, :3, :3].contiguous()


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    pos = x > 0
    if torch.is_grad_enabled():
        ret[pos] = torch.sqrt(x[pos])
    else:
        ret = torch.where(pos, torch.sqrt(x), ret)
    return ret


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    scale = 0.5 * torch.sinc(angles * 0.5 / torch.pi)
    return torch.cat([torch.cos(angles * 0.5), axis_angle * scale], dim=-1)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(q, -1)
    s = 2.0 / (q * q).sum(-1)
    o = torch.stack([
        1 - s*(j*j + k*k),  s*(i*j - k*r),      s*(i*k + j*r),
        s*(i*j + k*r),      1 - s*(i*i + k*k),  s*(j*k - i*r),
        s*(i*k - j*r),      s*(j*k + i*r),      1 - s*(i*i + j*j),
    ], -1)
    return o.reshape(q.shape[:-1] + (3, 3))


def standardize_quaternion(q: torch.Tensor) -> torch.Tensor:
    return torch.where(q[..., 0:1] < 0, -q, q)


def matrix_to_quaternion(mat: torch.Tensor) -> torch.Tensor:
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
    norms = torch.norm(q[..., 1:], p=2, dim=-1, keepdim=True)
    half  = torch.atan2(norms, q[..., :1])
    scale = 0.5 * torch.sinc(half / torch.pi)
    return q[..., 1:] / scale


def axis_angle_to_matrix(axis_angle: torch.Tensor, fast: bool = False) -> torch.Tensor:
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
    return {'X': 0, 'Y': 1, 'Z': 2}[letter]


def _angle_from_tan(axis, other_axis, data, horizontal, tait_bryan):
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
    c, s = torch.cos(angle), torch.sin(angle)
    one, zero = torch.ones_like(angle), torch.zeros_like(angle)
    flat = {'X': (one,zero,zero,zero,c,-s,zero,s,c),
            'Y': (c,zero,s,zero,one,zero,-s,zero,c),
            'Z': (c,-s,zero,s,c,zero,zero,zero,one)}[axis]
    return torch.stack(flat, -1).reshape(angle.shape + (3, 3))


def matrix_to_euler_angles(mat: torch.Tensor, convention: str) -> torch.Tensor:
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
    if euler.dim() == 0 or euler.shape[-1] != 3 or len(convention) != 3:
        raise ValueError(f"Invalid euler angles shape {euler.shape} or convention {convention}")
    mats = [_axis_angle_rotation(c, e) for c, e in zip(convention, torch.unbind(euler, -1))]
    return torch.matmul(torch.matmul(mats[0], mats[1]), mats[2])


def matrix_to_axis_angle(mat: torch.Tensor, fast: bool = False) -> torch.Tensor:
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
    return matrix_to_euler_angles(axis_angle_to_matrix(axis_angle), convention)


def euler2aa(euler: torch.Tensor, convention: str = 'XYZ') -> torch.Tensor:
    return matrix_to_axis_angle(euler_angles_to_matrix(euler, convention))


# ---------------------------------------------------------------------------
# Pose-level helpers
# ---------------------------------------------------------------------------

def remove_rotation_from_axis(full_pose: torch.Tensor, axis: int,
                               convention: str = 'XYZ') -> torch.Tensor:
    """Zero the global rotation around one axis in a batch of poses.

    Args:
        full_pose: (B, pose_dim) axis-angle poses; first 3 values are global orient.
        axis: Axis index to zero (0=X, 1=Y, 2=Z).
        convention: Euler-angle convention for decomposition.

    Returns:
        Modified pose tensor with the specified axis rotation removed.
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
    """Copy one axis of the global orientation from *additional* into *base*.

    Args:
        additional: (B, pose_dim) source of the axis value.
        base: (B, pose_dim) pose whose other axes are kept.
        axis: Axis index to copy (0=X, 1=Y, 2=Z).

    Returns:
        New pose tensor with the merged global orientation.
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

    Args:
        mesh_v: (N, V, 3) array of vertex positions.
        Rxyz:   (N, 3) or (3,) rotation angles in degrees.

    Returns:
        Rotated vertex array of the same shape.
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
