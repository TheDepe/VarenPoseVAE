# Script for generating and visualizing 3D pose samples using the pose prior.
#
# This script loads a pre-trained pose prior to generate random 3D poses,
# constructs corresponding 3D meshes, and optionally saves them to disk.
# The generated meshes are displayed using trimesh.
#
# Arguments:
#     --body_model_path (str): Path to the pretrained quadruped body model.
#     --checkpoint_path (str): Path to the model checkpoint.
#     --num_samples (int): Number of pose samples to generate (default: 3).
#     --save_samples (bool): If set, saves generated poses and meshes to disk.
#     --temperature (float): Controls the variation in sampled poses (default: 1.0).
#
# Outputs:
#     - If --save_samples is set:
#         - Poses are saved as 'samples.npy' in the 'samples' directory.
#         - Meshes are saved as .ply files in the same directory.
#     - The generated 3D models are displayed in a trimesh scene.
#
# Example usage:
# NOTE: Script requires the follow structure to run, due to directory structure.
#     python -m example_scripts.sample_VAE --num_samples 5 --save_samples --temperature 1.5
#
# Dependencies:
#     - torch
#     - argparse
#     - trimesh
#     - quadruped_poser.utils.example_utils (for helper functions)
#
# Author: Dennis Perrett

import torch
import argparse
import trimesh
from varen import VAREN as BodyModel
from quadruped_poser.utils.example_utils import (load_model, 
                                    generate_poses, 
                                    create_meshes, 
                                    save_samples)


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--body_model_path', type=str, default="/ssd-disk/data_ssd/VAREN/models/VAREN")
    parser.add_argument('--checkpoint_path', type=str, default="/ssd-disk/data_ssd/VAREN/models/VarenPoser/VarenPoser2_0.pth")
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--save_samples', action='store_true')
    parser.add_argument('--temperature', type=float, default=1.0)
    return parser.parse_args()



def main():
    """Main function that runs the pose generation pipeline."""
    args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    shape = torch.randn(args.num_samples, 39).to(device)
    std = torch.ones(39, device=device) * 0.5   # small variance for most
    std[:2] = 1.5                               # large variance for first two

    # scale accordingly
    shape = shape * std
    model = load_model(args.body_model_path, args.checkpoint_path, device)
    body_model = BodyModel(args.body_model_path, use_muscle_deformations=False).to(device)
    poses = generate_poses(model, args.num_samples, args.temperature, device)
    scene = create_meshes(body_model, poses, device, shape=shape)

    if args.save_samples:
        save_samples(poses, scene)

    #trimesh.Scene(scene).show()


if __name__ == "__main__":
    main()
