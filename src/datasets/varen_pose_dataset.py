import numpy as np
from tqdm import tqdm
import torch
from pathlib import Path
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from src.utils.rotation_tools import aa2matrot, matrot2aa, aa2euler, euler2aa


class VarenMoCapData(Dataset):
    """
    A PyTorch Dataset for loading motion capture data from pickle files.

    This dataset searches for all pose files with the suffix '_stageii.pkl' in the given 
    directory, extracts the full-body poses, and stores them as a NumPy array.

    Args:
        data_dir (str or Path): Path to the directory containing the motion capture data.
    
    Expected Dataset Structure:
        Not important. Will find all files with name pattern '*_stageii.pkl' anywhere 
        within the root (and sub) directory.

    Attributes:
        poses (np.ndarray): A concatenated array of all extracted poses, where each pose 
            is represented as an N x 111 matrix (excluding the first three global rotation 
            components).

    Raises:
        FileNotFoundError: If the specified data directory does not exist.

    Example:
        dataset = VarenMoCapData("path/to/data")
        first_pose = dataset[0]  # Access a single pose sample
    """
    def __init__(self, data_dir, **kwargs):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")
        
        pose_files = list(self.data_dir.rglob('*_stageii.pkl'))
        
        self.poses = []

        for file in tqdm(pose_files):
            
            data = np.load(file, allow_pickle=True)
            full_pose = torch.Tensor(data['fullpose'])
            full_pose = remove_rotation_from_axis(full_pose, axis=2).double().numpy()

            self.poses.append(full_pose)
            break
        self.poses = np.concatenate(self.poses, axis=0)

        print(f"Found {len(self)} poses, across {len(pose_files)} files.")

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        return self.poses[idx]
    
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