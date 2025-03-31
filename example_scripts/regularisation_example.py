# Script for generating and visualizing 3D pose samples using the VAREN model.
#
# This script loads a pre-trained VAREN model to generate random 3D poses, 
# constructs corresponding 3D meshes, and optionally saves them to disk. 
# The generated meshes are displayed using trimesh.
#
# Arguments:
#     --varen_model_path (str): Path to the pre-trained VAREN model.
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
#     python -m examples.sample_VAE --num_samples 5 --save_samples --temperature 1.5
#
# Dependencies:
#     - torch
#     - argparse
#     - trimesh
#     - src.utils.example_utils (for helper functions)
#
# Author: Dennis Perrett

import torch
import argparse
import trimesh

from src.utils.example_utils import (load_model, 
                                    generate_poses, 
                                    create_meshes, 
                                    save_samples)


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--varen_model_path', type=str, default="/home/dperrett/Documents/Data/VAREN/models/VAREN")
    parser.add_argument('--checkpoint_path', type=str, default="/home/dperrett/Documents/Data/Checkpoints/VarenPoser.pth")
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--save_samples', action='store_true')
    return parser.parse_args()



def main():
    """Main function that runs the pose generation pipeline."""
    args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(args.varen_model_path, args.checkpoint_path, device)
    NUM_JOINTS = model.body_model.NUM_JOINTS
    poses_input = (torch.rand(args.num_samples, NUM_JOINTS * 3, device=device) - 0.5) * .8
    regularised_poses = model(poses_input)['pose_body'].reshape(args.num_samples, -1)
    
    colours = (torch.rand(args.num_samples,3) * 255).byte().cpu().numpy()
    raw_meshs = create_meshes(model, poses_input, device, colours)
    final_meshs = create_meshes(model, regularised_poses, device, colours*.5)
    for mesh in final_meshs:
        mesh.vertices += [0,0,-2.5]
    

    trimesh.Scene(raw_meshs+final_meshs).show()


if __name__ == "__main__":
    main()
