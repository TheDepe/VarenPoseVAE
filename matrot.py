import numpy as np
import torch
from varen_poser.utils.rotation_tools import aa2matrot, matrot2aa
from scipy.spatial.transform import Rotation as R


def random_rotation_matrices(n):
    """Generate n random 3x3 rotation matrices."""
    rotations = R.random(n).as_matrix()  # Generate n random rotation matrices
    return rotations  # Shape (n, 3, 3)



def remove_rot_axis_from_pose(pose, axis, convention='xyz'):


    euler_angles = R.from_matrix(aa2matrot(torch.tensor(pose))).as_euler(convention, degrees=False)
    euler_angles[:,axis] = 0 # remove plane rotation

    pose = torch.Tensor(R.from_euler(convention, euler_angles, degrees=False).as_matrix())
    pose = matrot2aa(pose).numpy()
    return pose

n = 1


poses = matrot2aa(torch.tensor(random_rotation_matrices(n))).numpy()
print(poses)
poses = remove_rot_axis_from_pose(poses, 2)
print(poses)




