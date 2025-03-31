# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2020.12.12


import numpy as np
import torch
from .tgm_conversion import *
from torch.nn import functional as F
from scipy.spatial.transform import Rotation as R


def local2global_pose(local_pose, kintree):
    bs = local_pose.shape[0]

    local_pose = local_pose.view(bs, -1, 3, 3)

    global_pose = local_pose.clone()

    for jId in range(len(kintree)):
        parent_id = kintree[jId]
        if parent_id >= 0:
            global_pose[:, jId] = torch.matmul(global_pose[:, parent_id], global_pose[:, jId])

    return global_pose





def matrot2aa(pose_matrot):
    '''
    :param pose_matrot: Nx3x3
    :return: Nx3
    '''
    bs = pose_matrot.size(0)
    homogen_matrot = F.pad(pose_matrot, [0, 1])
    pose = rotation_matrix_to_angle_axis(homogen_matrot)
    return pose


def aa2matrot(pose):
    '''
    :param Nx3
    :return: pose_matrot: Nx3x3
    '''
    bs = pose.size(0)
    num_joints = pose.size(1) // 3
    pose_body_matrot = angle_axis_to_rotation_matrix(pose)[:, :3, :3].contiguous()  # .view(bs, num_joints*9)
    return pose_body_matrot



from typing import Union, List


def rotate_points_xyz(mesh_v: np.ndarray, Rxyz: Union[List[int], np.ndarray]):
    '''

    :param mesh_v: Nxnum_vx3
    :param Rxyz: Nx3 or 3 in degrees
    :return:
    '''
    if Rxyz is not None:
        Rxyz = list(Rxyz)
        Rxyz = np.repeat(np.array(Rxyz).reshape(1, 3), repeats=len(mesh_v), axis=0)

    mesh_v_rotated = []

    for fId in range(mesh_v.shape[0]):
        angle = np.radians(Rxyz[fId, 0])
        rx = np.array([
            [1., 0., 0.],
            [0., np.cos(angle), -np.sin(angle)],
            [0., np.sin(angle), np.cos(angle)]
        ])

        angle = np.radians(Rxyz[fId, 1])
        ry = np.array([
            [np.cos(angle), 0., np.sin(angle)],
            [0., 1., 0.],
            [-np.sin(angle), 0., np.cos(angle)]
        ])

        angle = np.radians(Rxyz[fId, 2])
        rz = np.array([
            [np.cos(angle), -np.sin(angle), 0.],
            [np.sin(angle), np.cos(angle), 0.],
            [0., 0., 1.]
        ])
        mesh_v_rotated.append(rz.dot(ry.dot(rx.dot(mesh_v[fId].T))).T)

    return np.array(mesh_v_rotated)


def tmat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: NxBx3x3 array of a batch of rotation matrices
            - t: NxBx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row

    bs = R.shape[0]

    return torch.cat([F.pad(R.view(-1, 3, 3), [0, 0, 0, 1]),
                      F.pad(t.view(-1, 3, 1), [0, 0, 0, 1], value=1)], dim=2).view(bs, -1, 4, 4)


def batch_rigid_transform(rot_mats, joints, parents):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)  # BxNx3X1

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transform_chain = [tmat(rot_mats[:, 0], rel_joints[:, 0])[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]], tmat(rot_mats[:, i], rel_joints[:, i])[:, 0])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    return posed_joints


def aa2euler(axis_angle, convention: str = "xyz"):
    return R.from_matrix(aa2matrot(axis_angle)).as_euler(convention, degrees=False)

def euler2aa(euler_angles, convention: str = "xyz"):
    return torch.Tensor(R.from_euler(convention, euler_angles, degrees=False).as_matrix())


def remove_rotation_from_axis(full_pose: torch.Tensor, axis: int, convention: str = 'xyz') -> torch.Tensor:
    """
    Removes rotation around a specified axis from the global orient of a full pose represented in axis-angle format.

    Args:
        full_pose (torch.Tensor): A tensor of shape (batch_size, pose_dim) representing full body pose,
                                  where the first 3 values correspond to global orientation in axis-angle format.
        axis (int): The index of the rotation axis to be zeroed (0 for X, 1 for Y, 2 for Z).
        convention (str, optional): The Euler angle convention used for conversion. Defaults to 'xyz'.

    Returns:
        torch.Tensor: The modified pose tensor with the specified axis rotation removed.
    """
    global_orient = full_pose[:, :3]
    euler_angles = aa2euler(global_orient) #R.from_matrix(aa2matrot(global_orient)).as_euler(convention, degrees=False)
    euler_angles[:,axis] = 0 # remove plane rotation

    global_orient = euler2aa(euler_angles) #torch.Tensor(R.from_euler(convention, euler_angles, degrees=False).as_matrix())
    global_orient = matrot2aa(global_orient)
    
    full_pose[:, :3] = global_orient
    return full_pose

def merge_global_orients_along_axis(additional: torch.Tensor, base: torch.Tensor, axis: int) -> torch.Tensor:
    """
    Merges the global orientation of the `additional` pose into `base` along a specified axis.

    Args:
        additional (torch.Tensor): Tensor of shape (batch_size, pose_dim) containing the additional pose.
        base (torch.Tensor): Tensor of shape (batch_size, pose_dim) containing the base pose.
        axis (int): The axis (0 for X, 1 for Y, 2 for Z) along which the global orientation is merged.

    Returns:
        torch.Tensor: A new tensor with the modified global orientation.
    """
    base = base.clone()
    base_global_orient = base[:, :3] # full_pose -> GO
    addi_global_orient = additional[:, 3] # full pose -> GO

    base_euler_angles = aa2euler(base_global_orient)
    addi_euler_angles = aa2euler(addi_global_orient)

    base_euler_angles[:, axis] = addi_euler_angles[:, axis]

    merged_global_orient = euler2aa(base_euler_angles)
    base[:, :3] = merged_global_orient

    return base