from __future__ import annotations

import torch
import trimesh
import numpy as np

from pathlib import Path
from quadruped_poser.models.pose_prior import QuadrupedPosePrior
from varen import VAREN as BodyModel


def load_model(body_model_path: str, checkpoint_path: str, device: str) -> QuadrupedPosePrior:
    """Load the pose prior model and restore weights from a checkpoint.

    Instantiates a :class:`QuadrupedPosePrior`, moves it to *device*, and
    loads the state-dict found at *checkpoint_path*.  The model is returned in
    evaluation mode with gradients disabled via ``eval()``.

    Args:
        body_model_path: Path to the quadruped body model file.  Currently
            passed through via kwargs and reserved for future use.
        checkpoint_path: Filesystem path to the ``.pth`` checkpoint file
            containing the saved state-dict.
        device: PyTorch device string, e.g. ``'cuda'`` or ``'cpu'``.

    Returns:
        :class:`QuadrupedPosePrior` instance with checkpoint weights loaded,
        placed on *device* and set to eval mode.
    """
    model = QuadrupedPosePrior().to(device).eval()
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt, strict=False)
    return model


def generate_poses(
    model: QuadrupedPosePrior,
    num_samples: int,
    temperature: float,
    device: str,
) -> torch.Tensor:
    """Sample a batch of poses from the pose prior.

    Calls the model's ``sample_poses`` method and reshapes the result into a
    flat per-pose vector.  The number of generated poses is printed to stdout
    and the tensor is moved to *device* before being returned.

    Args:
        model: A loaded and eval-mode :class:`QuadrupedPosePrior`.
        num_samples: Number of pose samples to draw.
        temperature: Scalar controlling sampling diversity; higher values
            produce more varied poses.
        device: PyTorch device string, e.g. ``'cuda'`` or ``'cpu'``.

    Returns:
        Float tensor of shape ``(num_samples, num_joints * 3)`` containing the
        sampled pose parameters, located on *device*.
    """
    poses = model.sample_poses(num_samples, temperature=temperature)['pose_body'].reshape(num_samples, -1)
    print(f"{poses.shape[0]} poses generated...")
    return poses.to(device)


def create_meshes(
        model: BodyModel,
        poses: torch.Tensor,
        device: str,
        colours: np.ndarray | list | None = None,
        shape: torch.Tensor | None = None) -> list:
    """Create 3-D trimesh objects from a batch of pose samples.

    Runs the body model forward pass to obtain vertex positions for every pose
    in *poses*, then wraps each set of vertices in a
    :class:`trimesh.Trimesh`.  Meshes are offset along the Y-axis so that they
    do not overlap when visualised together.

    Args:
        model: Instantiated quadruped body model used for the forward pass.
        poses: Pose parameter tensor of shape ``(N, num_joints * 3)``.  The
            first three columns are treated as the global orientation.
        device: PyTorch device string, e.g. ``'cuda'`` or ``'cpu'``.
        colours: Optional per-mesh RGB colours as an array-like of shape
            ``(N, 3)`` with values in ``[0, 255]``.  When ``None``, random
            colours are generated.
        shape: Optional shape (beta) coefficients of shape ``(N, 39)``.  When
            ``None``, a zero tensor is used.

    Returns:
        List of :class:`trimesh.Trimesh` objects, one per input pose, offset
        along the Y-axis for non-overlapping display.
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


def save_samples(poses: torch.Tensor, scene: list, output_folder: str = "samples") -> None:
    """Save generated pose arrays and mesh files to disk.

    Creates *output_folder* (including any missing parents), writes all poses
    to ``samples.npy``, and exports each mesh as a ``.ply`` file named
    ``sample_mesh_<i>.ply``.  Each mesh is centred at the origin before export.

    Args:
        poses: Generated poses tensor of shape ``(N, num_joints * 3)``; saved
            as a NumPy array via :func:`numpy.save`.
        scene: List of :class:`trimesh.Trimesh` meshes to export, one per
            pose sample.
        output_folder: Path to the directory where output files are written.
            Created automatically if it does not exist.

    Returns:
        None
    """
    out_folder = Path(output_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    np.save(out_folder / "samples.npy", poses.detach().cpu().numpy())

    for i, mesh in enumerate(scene):
        mesh.vertices -= mesh.vertices.mean(axis=0)
        mesh.export(out_folder / f"sample_mesh_{i}.ply")
