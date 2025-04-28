import torch
import trimesh
import numpy as np

from pathlib import Path
from src.models.varen_poser import VarenPoser
from typing import Union, List

def load_model(varen_model_path: str, checkpoint_path: str, device: str) -> VarenPoser:
    """Loads the VAREN model and checkpoint weights.
    
    Args:
        varen_model_path (str): Path to the VAREN model.
        checkpoint_path (str): Path to the model checkpoint.
        device (str): Device to load the model on ('cuda' or 'cpu').
    
    Returns:
        VarenPoser: Loaded VAREN model.
    """
    model = VarenPoser(varen_path=varen_model_path).to(device).eval()
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt, strict=False)
    return model


def generate_poses(model: VarenPoser, num_samples: int, temperature: float, device: str) -> torch.Tensor:
    """Generates pose samples using the VAREN model.
    
    Args:
        model (VarenPoser): The pre-trained VAREN model.
        num_samples (int): Number of pose samples to generate.
        temperature (float): Sampling diversity control.
        device (str): Device ('cuda' or 'cpu').
    
    Returns:
        torch.Tensor: Generated poses as a tensor.
    """
    poses = model.sample_poses(num_samples, temperature=temperature)['pose_body'].reshape(num_samples, -1)
    print(f"{poses.shape[0]} poses generated...")
    return poses.to(device)


def create_meshes(
        model: VarenPoser,
        poses: torch.Tensor,
        device: str,
        colours: Union[np.ndarray, List] = None,
        shape: torch.Tensor = None) -> list:
    """Creates 3D meshes from generated poses.
    
    Args:
        model (VarenPoser): The pre-trained VAREN model.
        poses (torch.Tensor): Pose samples.
        device (str): Device ('cuda' or 'cpu').
        colours (np.ndarray, List): Colours for each mesh.
    
    Returns:
        list: List of trimesh meshes representing the generated poses.
    """
    n_poses = poses.shape[0]
    if shape is None:
        shape = torch.zeros(n_poses, 39).to(device)

    #global_orient = torch.zeros(n_poses, 3).to(device)
    transl = torch.zeros(n_poses, 3).to(device)

    vertices = model.body_model(body_pose=poses[:,3:], betas=shape, transl=transl, global_orient=poses[:,:3]).vertices

    scene = []
    offset_step = 2.0

    if colours is None:
            colours = (torch.rand(n_poses,3) * 255).byte().cpu().numpy()

    for i, horse in enumerate(vertices):
        offset = np.array([0, i * offset_step, 0])
        horse_np = horse.detach().cpu().numpy() + offset
        mesh = trimesh.Trimesh(vertices=horse_np, faces=model.body_model.faces)
        mesh.visual.vertex_colors = np.tile(np.append(colours[i], 255), (horse_np.shape[0], 1))

        scene.append(mesh)

    return scene


def save_samples(poses: torch.Tensor, scene: list, output_folder: str = "samples"):
    """Saves pose samples and corresponding 3D meshes to disk.
    
    Args:
        poses (torch.Tensor): The generated poses.
        scene (list): List of trimesh meshes.
        output_folder (str, optional): Directory to save files. Defaults to "samples".
    """
    out_folder = Path(output_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    np.save(out_folder / "samples.npy", poses.detach().cpu().numpy())

    for i, mesh in enumerate(scene):
        mesh.vertices -= mesh.vertices.mean(axis=0)
        mesh.export(out_folder / f"sample_mesh_{i}.ply")