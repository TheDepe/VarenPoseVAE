# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# Code Developed by:
# Dennis Perrett <https://github.com/TheDepe>
#
# Based on the original implementation by
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2020.12.12


import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from varen import VAREN

from .model_components import BatchFlatten
from src.utils.angle_continuous_repres import geodesic_loss_R
from src.utils.rotation_tools import (
    matrot2aa,
    aa2matrot,
    remove_rotation_from_axis,
    merge_global_orients_along_axis
)



class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


class NormalDistDecoder(nn.Module):
    def __init__(self, num_feat_in, latentD):
        super(NormalDistDecoder, self).__init__()

        self.mu = nn.Linear(num_feat_in, latentD)
        self.logvar = nn.Linear(num_feat_in, latentD)

    def forward(self, Xout):
        return torch.distributions.normal.Normal(
            self.mu(Xout), F.softplus(self.logvar(Xout))
            )


class VarenPoser(nn.Module):
    """VAE-base Horse-Pose prior.
    
    VarenPoser is a Variational Autoencoder (VAE)-based model for learning 
    horse body poses. It consists of an encoder that compresses body poses 
    into a latent space and a decoder that reconstructs poses from the 
    latent representation.

    Attributes:
        num_joints (int): The number of joints in the horse body model.
        latentD (int): Dimensionality of the latent space.
        encoder_net (nn.Sequential): The network for encoding poses.
        decoder_net (nn.Sequential): The network for decoding poses.
    """
    def __init__(self, **kwargs):
        """
        Initialise the VarenPoser model.

        Args:
            varen_path (str): Path to the pre-trained VAREN body model.
            **kwargs: Additional arguments for potential extensions.
        """
        super(VarenPoser, self).__init__()

        num_neurons, self.latentD = 512, 16 #32

        self.num_joints = 37 + 1 # for global orient
        n_features = self.num_joints * 3

        self.encoder_net = nn.Sequential(
            BatchFlatten(),
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, num_neurons),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_neurons),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.Linear(num_neurons, num_neurons),
            NormalDistDecoder(num_neurons, self.latentD)
        )

        self.decoder_net = nn.Sequential(
            nn.Linear(self.latentD, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.LeakyReLU(),
            nn.Linear(num_neurons, self.num_joints * 6),
            ContinousRotReprDecoder(),
        )

    def encode(self, pose_body):
        """
        Encodes the input pose into a probabilistic latent representation.

        Args:
            pose_body (torch.Tensor): Input pose of shape (N, num_joints * 3).

        Returns:
            Normal: A normal distribution in latent space representing the pose.
        """
        return self.encoder_net(pose_body)

    def decode(self, Z):
        """
        Decodes a latent representation into a reconstructed pose.

        Args:
            Z (torch.Tensor): Latent representation of shape (N, latentD).

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'pose_body' (torch.Tensor): Decoded pose in axis-angle
                    format.
                - 'pose_body_matrot' (torch.Tensor): Decoded pose in matrix
                    rotation format.
        """
        bs = Z.shape[0]

        prec = self.decoder_net(Z)

        return {
            'pose_body': matrot2aa(prec.view(-1, 3, 3)).view(bs, -1, 3),
            'pose_body_matrot': prec.view(bs, -1, 9)
        }


    def forward(self, pose_body):
        """
        Performs a forward pass through the encoder and decoder.

        Args:
            pose_body (torch.Tensor): Input body pose of shape
                (N, num_joints * 3).

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing decoded pose
                representations.
        """

        q_z = self.encode(pose_body)
        q_z_sample = q_z.rsample()
        decode_results = self.decode(q_z_sample)
        decode_results.update(
            {'poZ_body_mean': q_z.mean, 'poZ_body_std': q_z.scale, 'q_z': q_z}
            )
        return decode_results

    def sample_poses(self, num_poses, temperature=1.0, seed=None):
        """
        Samples new poses from the latent space.

        Args:
            num_poses (int): Number of poses to generate.
            seed (int, optional): Random seed for reproducibility.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing generated poses.
        """
        np.random.seed(seed)

        assert temperature > 0.0, (
            f"Temperature must be positive and non-zero. Got {temperature}"
        )
        
        # Get model's data type & device
        some_weight = next(self.parameters())  
        dtype = some_weight.dtype
        device = some_weight.device

        self.eval()
        with torch.no_grad():
            Zgen = torch.tensor(
                np.random.normal(
                    0.,
                    1.0 * temperature,
                    size=(num_poses, self.latentD)),
                    dtype=dtype,
                    device=device
                )

        return self.decode(Zgen)
    
    def regularise_pose(self, full_pose: torch.Tensor) -> torch.Tensor:
        """
        Regularizes the pose using the VarenPoser model.

        Args:
            full_pose (Tensor): A tensor of shape (batch_size, pose_dim)
            representing full body pose.

        Returns:
            Tensor: The regularized pose.
        """

        # remove go z rotation
        prepared_pose = remove_rotation_from_axis(full_pose, axis=2)

        regularised_pose = self(prepared_pose)['pose_body']

        # add original z rotation back
        output_pose = merge_global_orients_along_axis(
            full_pose, regularised_pose, axis=2
            )

        return output_pose
    
    
class VarenPoserTrainingExtension(VarenPoser):
    """
    Extended version of VarenPoser with additional training utilities.

    Includes functions for computing loss and constructing meshes.

    Attributes:
        body_model (VAREN): A pre-trained horse model used for pose
        reconstruction.
    Methods:
        construct_meshes(dorig, drec): Constructs original and reconstructed
            3D meshes.
        compute_loss(dorig, drec): Computes loss terms including reconstruction
            and KL divergence loss.
    """
    def __init__(self, varen_path, **kwargs):
        """
        Initializes the extended VarenPoser class.

        Args:
            varen_path (str): Path to the pre-trained VAREN model.
            **kwargs: Additional arguments.
        """
        super().__init__(varen_path, **kwargs)
        self.body_model = VAREN(varen_path)
        # Dont' train body model
        for param in self.body_model.parameters():
            param.requires_grad = False

    def construct_meshes(self, dorig, drec):
        """
        Constructs 3D meshes for original and reconstructed poses.

        Args:
            dorig (torch.Tensor): Original pose tensor of shape
            (N, num_joints * 3).
            drec (Dict[str, torch.Tensor]): Dictionary of reconstructed pose
                tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Original and reconstructed mesh
                objects.
        """
        bs = dorig.shape[0]
        betas = torch.zeros(bs,39).to(dorig.device).float()
        transl = torch.zeros(bs,3).to(dorig.device).float()
        
        full_pose_in = dorig.float()
        full_pose_pred = drec['pose_body'].reshape(bs,-1).float()
        go_in = full_pose_in[:, :3]
        go_pred = full_pose_pred[:, :3]

        pose_in = full_pose_in[:, 3:]
        pose_pred = full_pose_pred[:, 3:]

        mesh_orig = self.body_model(
            global_orient=go_in,
            body_pose=pose_in,
            transl=transl,
            betas=betas
            )
        mesh_recon = self.body_model(
            global_orient=go_pred,
            body_pose=pose_pred,
            transl=transl,
            betas=betas
            )

        return mesh_orig, mesh_recon

    def compute_loss(self, pose_gt, pose_pred):
        """
        Computes the total loss for training.

        Loss components include:
        - KL Divergence
        - L1 Mesh Reconstruction Loss
        - Geodesic Rotation Loss
        - Joint Position Loss

        Args:
            pose_gt (torch.Tensor): Original body pose tensor of shape
                (N, num_joints * 3).
            drepose_predc (Dict[str, torch.Tensor]): Dictionary containing
                reconstructed poses.

        Returns:
            Dict[str, Dict[str, torch.Tensor]]: Dictionary containing weighted
            and unweighted loss terms.
        """
        l1_loss = torch.nn.L1Loss(reduction='mean')
        geodesic_loss = geodesic_loss_R(reduction='mean')

        bs, latentD = pose_pred['poZ_body_mean'].shape
        device = pose_pred['poZ_body_mean'].device

        loss_kl_wt = 0.005
        loss_rec_wt = 4
        loss_matrot_wt = 2
        loss_jtr_wt = 2

        q_z = pose_pred['q_z']

        # Reconstruction loss - L1 on the output mesh
        mesh_orig, mesh_recon = self.construct_meshes(pose_gt, pose_pred)
        v2v = l1_loss(mesh_orig.vertices, mesh_recon.vertices)

        # KL loss
        p_z = torch.distributions.normal.Normal(
            loc=torch.zeros((bs, latentD), device=device, requires_grad=False),
            scale=torch.ones(
                (bs, latentD), device=device, requires_grad=False)
                )
        weighted_loss_dict = {
            'loss_kl':loss_kl_wt * torch.mean(
                torch.sum(
                    torch.distributions.kl.kl_divergence(q_z, p_z),
                    dim=[1]
                    )),
            'loss_mesh_rec': loss_rec_wt * v2v
        }

        weighted_loss_dict['matrot'] = loss_matrot_wt * geodesic_loss(
            pose_pred['pose_body_matrot'].view(-1,3,3).double(),
            aa2matrot(pose_gt.view(-1,3)))
        weighted_loss_dict['jtr'] = loss_jtr_wt * l1_loss(
            mesh_orig.joints,
            mesh_recon.joints
            )

        weighted_loss_dict['loss_total'] = torch.stack(
            list(weighted_loss_dict.values())
            ).sum()

        return {'weighted_loss': weighted_loss_dict, 'unweighted_loss': {}} 