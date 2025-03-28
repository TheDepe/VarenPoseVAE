# Script for generating and visualizing the latent space of the VAE.
#
# The VAE latent space is a 32-dimensional standard normal distribution.
# This script explores the effect of each latent dimension by creating input vectors 
# where each dimension is individually set to -2, 0, and +2 standard deviations (SD), 
# while all other dimensions remain at 0. This helps visualize the influence of 
# each latent dimension on the generated 3D horse poses.
#
# The script generates 96 different poses (32 dimensions * 3 values per dimension), 
# converts them into 3D meshes, and saves them with names indicating which 
# dimension and value were used (e.g., dim0_2neg.ply, dim0_0.ply, dim0_2pos.ply).
#
# Outputs:
#     - Poses are stored as .ply files in the 'samples' directory.
#     - Each filename follows the pattern: dimX_Val.ply
#         where:
#             - X is the latent dimension index (0 to 31).
#             - Val is either "2neg" (-2SD), "0" (0SD), or "2pos" (+2SD).
#
# Example usage:
# NOTE: Script requires the follow structure to run, due to directory structure. 
#     python -m examples.visualize_latent_space.py --save_samples
#
# Dependencies:
#     - torch
#     - argparse
#     - trimesh
#     - numpy
#     - pathlib
#     - src.utils.example_utils (for helper functions)
#
#
# Author: Dennis Perrett

import torch
import argparse
import trimesh
import numpy as np

from pathlib import Path

from src.utils.example_utils import load_model


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--varen_model_path', type=str, default="/home/dperrett/Documents/Data/VAREN/models/VAREN")
    parser.add_argument('--checkpoint_path', type=str, default="/home/dperrett/Documents/Data/Checkpoints/VarenPoser.pth")
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--save_samples', action='store_true')
    parser.add_argument('--temperature', type=float, default=1.0)
    return parser.parse_args()


def sample_poses_custom(model, temperature=1.0, sd=1.0, seed=None):
    """
    Samples new poses by setting each latent dimension to -2sd, 0, or +2sd, 
    keeping all other dimensions fixed to 0. For each latent dimension, 
    the three possibilities are considered (-2sd, 0, +2sd).
    
    Args:
        model (torch.nn.Module): The pre-trained model used to decode latent vectors into poses.
        temperature (float, optional): Scale factor for the randomness of latent vectors. Default is 1.0.
        sd (float, optional): The standard deviation used to create latent vectors. Default is 1.0.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing generated poses.
    """
    np.random.seed(seed)
    
    assert temperature > 0.0, f"Temperature must be positive and non-zero. Got {temperature}"
    assert sd > 0.0, f"Standard deviation must be positive. Got {sd}"

    # Define the latent values as [-2sd, 0, +2sd] for each dimension
    latent_values = torch.tensor([-2 * sd, 0.0, 2 * sd], dtype=torch.float32)

    latent_vectors = []

    # Create one vector for each latent dimension
    for i in range(model.latentD):
    
        latent_vector = torch.zeros(model.latentD, dtype=torch.float32)
        for val in latent_values:
            latent_vector[i] = val

            latent_vectors.append(latent_vector.clone())

    latent_tensor = torch.stack(latent_vectors)

    device = next(model.parameters()).device
    latent_tensor = latent_tensor.to(device)

    model.eval()
    with torch.no_grad():
        return model.decode(latent_tensor)

def visualize_and_export_horses(poses, model, out_folder="samples"):
    """
    Visualize and export the 96 horses as trimesh objects with naming convention based on latent dimensions.
    
    Args:
        poses (torch.Tensor): Tensor containing the generated poses with shape (96, 37, 3).
        model (nn.Module): The model used to generate the poses, used for mesh creation.
        out_folder (str): Directory where the meshes will be saved.
    """
    # Create the output folder if it doesn't exist
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    
    # Loop through each latent dimension
    latentD = poses.shape[1]  # Should be 32
    num_poses = poses.shape[0]  # Should be 96

    # Generate the 3D mesh for the modified pose
    shape = torch.zeros(num_poses, 39).to(poses.device)  
    global_orient = torch.zeros(num_poses, 3).to(poses.device)  
    transl = torch.zeros(num_poses, 3).to(poses.device)  

    vertices = model.body_model(body_pose=poses, betas=shape, transl=transl, global_orient=global_orient).vertices

    file_names = []

    latent_values = ["2neg", "0", "2pos"]
    for i in range(latentD):
        for val in latent_values:
            file_names.append(f"dim{i}_{val}")

    # Create and export a mesh for each modified pose
    for file_name, horse in zip(file_names,vertices):
        horse_np = horse.detach().cpu().numpy()

        mesh = trimesh.Trimesh(vertices=horse_np, faces=model.body_model.faces)
        mesh.export(out_folder / f"{file_name}.ply")
        print(f"Exported: {file_name}.ply")

def main():
    """Main function that runs the pose generation pipeline."""
    args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(args.varen_model_path, args.checkpoint_path, device)

    poses = sample_poses_custom(model, temperature=1.0, sd=1.0, seed=None)['pose_body']
    poses = poses.reshape(poses.shape[0], -1)
 
    visualize_and_export_horses(poses, model)


if __name__ == "__main__":
    main()
