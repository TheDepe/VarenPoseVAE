import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset


class VarenAlignmentData(Dataset):
    def __init__(self, data_dir, split="train", **kwargs):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")
        
        pose_files = list(self.data_dir.rglob('*_pose.npy'))
        
        self.poses = []
        for file in tqdm(pose_files):
            data = np.load(file, allow_pickle=True).item()
            pose = data['pose'].astype(np.float64)
            

            # CORRECT JAW:
            pose[35*3:35*3+3] = 0
            self.poses.append(pose)
        

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        return self.poses[idx]



class VarenMoCapData(Dataset):
    def __init__(self, data_dir, split="train", **kwargs):
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