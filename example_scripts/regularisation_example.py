# Script for regularising and visualizing 3D pose samples using the pose prior.
#
# This script loads a pre-trained pose prior, regularises random input poses,
# constructs before/after 3D meshes, and displays them using trimesh.
#
# Arguments:
#     --body_model_path (str): Path to the pretrained quadruped body model.
#     --checkpoint_path (str): Path to the model checkpoint.
#     --num_samples (int): Number of pose samples to generate (default: 3).
#     --save_samples (bool): If set, saves generated poses and meshes to disk.
#
# Outputs:
#     - Raw and regularised meshes displayed side-by-side in a trimesh scene.
#
# Example usage:
# NOTE: Script requires the follow structure to run, due to directory structure.
#     python -m example_scripts.regularisation_example --num_samples 5
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
    parser.add_argument('--body_model_path', type=str, default="/home/dperrett/Documents/Data/VAREN/models/VAREN")
    parser.add_argument('--checkpoint_path', type=str, default="/home/dperrett/Documents/Data/Checkpoints/VarenPoser.pth")
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--save_samples', action='store_true')
    return parser.parse_args()



def main():
    """Main function that runs the pose generation pipeline."""
    args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    body_model = BodyModel(args.body_model_path).to(device)
    model = load_model(args.body_model_path, args.checkpoint_path, device)
    NUM_JOINTS = model.num_joints
    poses_input = (torch.rand(args.num_samples, NUM_JOINTS * 3, device=device) - 0.5) * .8
    regularised_poses = model(poses_input)['pose_body'].reshape(args.num_samples, -1)
    
    colours = (torch.rand(args.num_samples,3) * 255).byte().cpu().numpy()
    raw_meshs = create_meshes(body_model, poses_input, device, colours)
    final_meshs = create_meshes(body_model, regularised_poses, device, colours*.5)
    for mesh in final_meshs:
        mesh.vertices += [0,0,-2.5]
    

    trimesh.Scene(raw_meshs+final_meshs).show()


if __name__ == "__main__":
    main()
