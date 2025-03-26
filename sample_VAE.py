import torch
import argparse
import trimesh

import numpy as np
from src.models.vposer import VPoser
from src.utils.logger import *



# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--varen_model_path', type=str, default="/home/dperrett/Documents/Data/VAREN/models/VAREN")
parser.add_argument('--num_samples', type=int, default="3")
parser.add_argument('--save_samples', action='store_true')

args = parser.parse_args()



device = 'cuda'

model = VPoser(input_dim=1 ,latent_dim=128).to(device).eval()
epoch = 31
ckpt = torch.load(f"/home/dperrett/Documents/horse_project/Repos/VarenPoseVAE/logs/2025_03_26__08_50_28/VAREN_pose_VAE_epoch_{epoch}.pth", weights_only=False)

model.load_state_dict(ckpt, strict=False)


poses = model.sample_poses(args.num_samples)['pose_body'].reshape(args.num_samples, -1)

print(f"{poses.shape[0]} poses generated...")


shape = torch.zeros(args.num_samples, 39).to(device)
global_orient = torch.zeros(args.num_samples, 3).to(device)
transl = torch.zeros(args.num_samples, 3).to(device)
vertices = model.body_model(body_pose=poses, betas=shape, transl=transl, global_orient=global_orient).vertices



scene = []
offset_step = 2.0  # Adjust spacing between horses




for i, horse in enumerate(vertices):
     offset = np.array([0, i * offset_step, 0])  # Move each horse along y-axis
     horse_np = horse.detach().cpu().numpy() + offset  # Apply offset

     num_faces = model.body_model.faces.shape[0]  # Get number of faces
     num_vertices = horse_np.shape[0]  # Get number of vertices

     # Generate a random color and ensure correct data type
     mesh_colour = (torch.rand(3) * 255).byte().cpu().numpy()
     face_colors = np.tile(np.append(mesh_colour, 255), (num_faces, 1))  # Add alpha (RGBA)

     # Create mesh
     mesh = trimesh.Trimesh(vertices=horse_np, faces=model.body_model.faces)
    
     # Apply colors to vertices
     mesh.visual.vertex_colors = np.tile(np.append(mesh_colour, 255), (num_vertices, 1))

     scene.append(mesh)

if args.save_samples:
    out_folder = Path("samples")
    out_folder.mkdir(parents=True, exist_ok=True)  # Corrected mkdir


    poses_np = poses.detach().cpu().numpy()
    np.save(out_folder / "samples.npy", poses_np)


    for i, mesh in enumerate(scene):
        centroid = mesh.vertices.mean(axis=0)  # Compute centroid
        mesh.vertices -= centroid  # Center the mesh
        mesh.export(out_folder / f"sample_mesh_{i}.ply")

# Show the scene
trimesh.Scene(scene).show()

