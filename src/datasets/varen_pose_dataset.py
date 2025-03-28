import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset


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
            pose = data['fullpose'].astype(np.float64)[:, 3:] # N x 113

            self.poses.append(pose)

        self.poses = np.concatenate(self.poses, axis=0)

        print(f"Found {len(self)} poses, across {len(pose_files)} files.")
    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        return self.poses[idx]