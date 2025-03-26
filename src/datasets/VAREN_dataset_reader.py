import json
import torch
import trimesh
import numpy as np
import kaolin as kal

from pathlib import Path


# Base Class for traversing Dataset structure
class HorseData(torch.utils.data.Dataset):

    def __init__(self, data_dir, method, split="train", **kwargs):

        self.data_dir = Path(data_dir)
        self.method = method
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")
        
        # Set any additional keyword arguments used for subclasses
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Set which data to read from
        if self.method in ["fitting","alignment"]:
            self.metadata_dir = "alignment_metadata"
        elif self.method == "muscle_training":
            self.metadata_dir = "muscle_training_metadata"
        elif self.method == "shape_training":
            self.metadata_dir = "shape_space_training_metadata"
        else:
            raise NotImplementedError(f"Dataloading for {self.method} method not implemented")
        

        # Define data split. These are set by the preprocess_dataset.py script.
        if split=="train":
            file = "train.txt"
        elif split=="test":
            file = "test.txt"
        elif split=="val":
            file = "val.txt"
        else:
            file = f"{split}.txt" # Custom file for eg test runs or overfitting etc

        
        metadata_path = self.data_dir / "METADATA" / self.metadata_dir / file
        with open(metadata_path) as f:
            self.data = [line.strip() for line in f if not line.lstrip().startswith("#")]


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        raise NotImplementedError


# Class for returning Data for training muscle deformations
class HorseMuscleData(HorseData):
    def __init__(self, data_dir, split="train", **kwargs):
        super().__init__(data_dir, method="muscle_training", split=split, **kwargs)

    def __getitem__(self, index):
        data = str(self.data_dir / self.data[index])
        # Load Registration Data. Indexing originally done by Silvia's Registration Script
        registration_data = np.load(data+"_pose.npy", allow_pickle=True).item() # Flattened Array
        
        mesh = trimesh.load_mesh(data+"_scan.ply")
        verts = torch.tensor(mesh.vertices, dtype=torch.float32).unsqueeze(0) # needs batch index
        faces = torch.tensor(mesh.faces, dtype=torch.long) # doesn't need batch index
        verts, _ = kal.ops.mesh.sample_points(verts, faces, num_samples=20000) # remove batch index

        
        return {
            "transl": torch.tensor(registration_data['transl'], dtype=torch.float32),
            "pose": torch.tensor(registration_data['pose'], dtype=torch.float32), 
            "betas": torch.tensor(registration_data['betas'], dtype=torch.float32), 
            "global_orient": torch.tensor(registration_data['global_orient'], dtype=torch.float32),
            "scan_vertices": verts.squeeze().to(torch.float32)
            }

# Class for returning Data for Alignment
class HorseAlignmentData(HorseData):
    """
    TODO: Documentation is missing
    """
    def __init__(self, data_dir, split="train", **kwargs):
        super().__init__(data_dir, method="alignment", split=split, **kwargs)

    def __getitem__(self, index):
        data = str(self.data_dir / self.data[index])

        mesh = trimesh.load_mesh(data+"_scan.ply")
        mesh.vertices = mesh.vertices / self.scaling
        
        if self.centered:
            mesh.vertices -= mesh.vertices.mean(0)
        
        try:
            params = np.load(data+"_pose.npy", allow_pickle=True).item()
        except FileNotFoundError:
            params = {
                    "transl": torch.full((3,), float('nan')),
                    "pose": torch.full((111,), float('nan')),
                    "global_orient": torch.full((3,), float('nan')),
                    "betas": torch.full((39,), float('nan'))
                    }

        # NOTE: We use a custom collate function for handling the trimesh object. 
        # If using multiple meshes, it samples each to ensure shape consistency
        item = {
            "file_name": self.data[index],
            "pcd": mesh
        }


        if params is not None and self.use_existing_params:
            item['params'] = {k: torch.tensor(v).to(torch.float32) for k,v in params.items()}


        # Include keypoints if not ignored
        if not self.ignore_keypoints:
            kp_path = Path(data + "_keypoints.json")
            try:
                with kp_path.open('r') as file:
                    keypoints = json.load(file)
                    # Get 3D locations of keypoints
                    keypoints = {key: mesh[val] for key ,val in keypoints.items()}
                item['keypoints'] = keypoints
            except (FileNotFoundError, json.JSONDecodeError) as e:
                item['keypoints'] = {}


        return item


class HorseShapeSpaceData(HorseData):
    pass


if __name__ == "__main__":
    data = HorseAlignmentData("/home/dperrett/Documents/Data/VAREN/VARENset", split="train", scaling=1000.0, centered=True, ignore_keypoints=False)


    print(data[0])