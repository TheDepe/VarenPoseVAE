import torch
import trimesh
import numpy as np

from pathlib import Path
from varen_poser.models.pose_prior import QuadrupedPosePrior
from typing import Union, List
from varen import VAREN


def load_model(body_model_path: str, checkpoint_path: str, device: str) -> QuadrupedPosePrior:
    """Loads the pose prior and its checkpoint weights.

    Args:
        body_model_path: Path to the quadruped body model (passed through via kwargs).
        checkpoint_path: Path to the model checkpoint (.pth file).
        device: Device string ('cuda' or 'cpu').

    Returns:
        Loaded QuadrupedPosePrior in eval mode.
    """
    model = QuadrupedPosePrior().to(device).eval()
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt, strict=False)
    return model


def generate_poses(model: QuadrupedPosePrior, num_samples: int, temperature: float, device: str) -> torch.Tensor:
    """Samples poses from the pose prior.

    Args:
        model: A loaded QuadrupedPosePrior.
        num_samples: Number of poses to generate.
        temperature: Sampling diversity control.
        device: Device string ('cuda' or 'cpu').

    Returns:
        Generated poses as a tensor of shape (num_samples, num_joints * 3).
    """
    poses = model.sample_poses(num_samples, temperature=temperature)['pose_body'].reshape(num_samples, -1)
    print(f"{poses.shape[0]} poses generated...")
    return poses.to(device)


def create_meshes(
        model: VAREN,
        poses: torch.Tensor,
        device: str,
        colours: Union[np.ndarray, List] = None,
        shape: torch.Tensor = None) -> list:
    """Creates 3D meshes from pose samples using a body model.

    Args:
        model: Quadruped body model used for mesh construction.
        poses: Pose tensor of shape (N, num_joints * 3).
        device: Device string ('cuda' or 'cpu').
        colours: Optional per-mesh RGB colours, shape (N, 3).
        shape: Optional shape (beta) coefficients, shape (N, 39).

    Returns:
        List of trimesh.Trimesh objects, one per pose.
    """
    n_poses = poses.shape[0]
    if shape is None:
        shape = torch.zeros(n_poses, 39).to(device)

    transl = torch.zeros(n_poses, 3).to(device)
    vertices = model(
        body_pose=poses[:, 3:],
        betas=shape,
        transl=transl,
        global_orient=poses[:, :3],
    ).vertices

    if colours is None:
        colours = (torch.rand(n_poses, 3) * 255).byte().cpu().numpy()

    scene = []
    offset_step = 2.0
    for i, verts in enumerate(vertices):
        offset = np.array([0, i * offset_step, 0])
        verts_np = verts.detach().cpu().numpy() + offset
        mesh = trimesh.Trimesh(vertices=verts_np, faces=model.faces)
        mesh.visual.vertex_colors = np.tile(np.append(colours[i], 255), (verts_np.shape[0], 1))
        scene.append(mesh)

    return scene


def save_samples(poses: torch.Tensor, scene: list, output_folder: str = "samples"):
    """Saves pose arrays and mesh files to disk.

    Args:
        poses: Generated poses tensor.
        scene: List of trimesh.Trimesh meshes.
        output_folder: Directory to write output files into.
    """
    out_folder = Path(output_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    np.save(out_folder / "samples.npy", poses.detach().cpu().numpy())

    for i, mesh in enumerate(scene):
        mesh.vertices -= mesh.vertices.mean(axis=0)
        mesh.export(out_folder / f"sample_mesh_{i}.ply")
