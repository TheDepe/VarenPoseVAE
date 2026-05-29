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


from typing import Dict, Optional

import numpy as np
import torch
from hbm import BodyModel
from torch import nn
from torch.nn import functional as F
from varen_poser.utils.angle_continuous_repres import geodesic_loss_R
from varen_poser.utils.rotation_tools import (
    aa2matrot,
    matrot2aa,
    merge_global_orients_along_axis,
    remove_rotation_from_axis,
)

from .model_components import BatchFlatten


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


class BasePoseMapper(nn.Module):
    """Abstract base class for pose mappers.

    Defines the interface that all mappers must implement. The forward()
    method is intentionally not used — VarenPoser calls to_canonical() and
    from_canonical() directly so that encode/decode remain mapper-agnostic.
    """

    def to_canonical(
        self, source_pose: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Map a source-layout pose to the canonical VarenPoser layout.

        Args:
            source_pose (torch.Tensor): Input pose in source layout.

        Returns:
            Dict with:
              - 'canonical_pose': (N, canonical_num_joints, 3)
              - 'extra_pose':     (N, E, 3) or None

        """
        raise NotImplementedError

    def from_canonical(
        self,
        canonical_pose: torch.Tensor,
        extra_pose: Optional[torch.Tensor],
        source_pose_original: torch.Tensor,
    ) -> torch.Tensor:
        """Map a canonical-layout pose back to the source layout.

        Args:
            canonical_pose (torch.Tensor): Decoded pose in canonical layout.
            extra_pose (torch.Tensor | None): Extra joints stashed by
                to_canonical(), to be reinserted unchanged.
            source_pose_original (torch.Tensor): The original source pose,
                used to preserve extra-joint values.

        Returns:
            torch.Tensor: Pose in source layout (N, num_source_joints, 3).

        """
        raise NotImplementedError


class IdentityMapper(BasePoseMapper):
    """A no-op mapper for when source and canonical layouts are identical.

    Used as the default when no mapper is provided to VarenPoser. Poses pass
    through unchanged in both directions.

    Args:
        num_joints (int): Number of joints. Must match VarenPoser's canonical
            num_joints (default: 38).
        values_per_joint (int): Values per joint (default: 3 for axis-angle).

    """

    def __init__(self, num_joints: int = 38, values_per_joint: int = 3):
        super().__init__()
        self.num_joints = num_joints
        self.vpj = values_per_joint

    def to_canonical(
        self, source_pose: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {
            "canonical_pose": self._ensure_3d(source_pose),
            "extra_pose": None,
        }

    def from_canonical(
        self,
        canonical_pose: torch.Tensor,
        extra_pose: Optional[torch.Tensor],
        source_pose_original: torch.Tensor,
    ) -> torch.Tensor:
        return self._ensure_3d(canonical_pose)

    def _ensure_3d(self, pose: torch.Tensor) -> torch.Tensor:
        if pose.dim() == 2:
            return pose.reshape(pose.shape[0], self.num_joints, self.vpj)
        return pose

    def extra_repr(self) -> str:
        return f"num_joints={self.num_joints}, values_per_joint={self.vpj} (identity/no-op)"


class VarenPoser(nn.Module):
    """VAE-based Horse-Pose prior.

    VarenPoser is a Variational Autoencoder (VAE)-based model for learning
    horse body poses. It consists of an encoder that compresses body poses
    into a latent space and a decoder that reconstructs poses from the
    latent representation.

    An optional pose mapper can be attached so that poses from a different
    joint configuration are transparently remapped before encoding and
    remapped back after decoding. If no mapper is provided, an IdentityMapper
    is used (source layout assumed to match canonical layout).

    Attributes:
        num_joints (int): The number of joints in the canonical horse body model.
        latentD (int): Dimensionality of the latent space.
        encoder_net (nn.Sequential): The network for encoding poses.
        decoder_net (nn.Sequential): The network for decoding poses.
        pose_mapper (BasePoseMapper): Maps between source and canonical layouts.

    """

    def __init__(self, pose_mapper: Optional[BasePoseMapper] = None, **kwargs):
        """Initialise the VarenPoser model.

        Args:
            pose_mapper (BasePoseMapper, optional): A BasePoseMapper subclass
                instance (e.g. PoseMapping). If None, an IdentityMapper is
                used — i.e. source layout is assumed to match canonical layout.
                Can also be set or swapped later via with_mapper().
            **kwargs: Additional arguments for potential extensions.

        """
        super(VarenPoser, self).__init__()

        num_neurons, self.latentD = 512, 32

        self.num_joints = 37 + 1  # body joints + global orient
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
            NormalDistDecoder(num_neurons, self.latentD),
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

        self.pose_mapper = (
            pose_mapper
            if pose_mapper is not None
            else IdentityMapper(num_joints=self.num_joints)
        )

    def with_mapper(self, pose_mapper: BasePoseMapper) -> "VarenPoser":
        """Attach or replace the pose mapper (builder/fluent pattern).

        Allows the mapper to be set or swapped after construction::

            model = VarenPoser().with_mapper(
                PoseMapping(joint_index_map=[...], num_source_joints=50)
            )

        Args:
            pose_mapper (BasePoseMapper): The mapper to attach.

        Returns:
            self, so calls can be chained.

        """
        self.pose_mapper = pose_mapper
        return self

    def encode(self, pose_body):
        """Encodes a canonical-layout pose into a latent distribution.

        Always operates in canonical space — mapping is handled by forward().

        Args:
            pose_body (torch.Tensor): Input pose of shape (N, num_joints * 3),
                in canonical layout.

        Returns:
            Normal: A normal distribution in latent space representing the pose.

        """
        return self.encoder_net(pose_body)

    def decode(self, Z):
        """Decodes a latent representation into a canonical-layout pose.

        Always operates in canonical space — mapping is handled by forward().

        Args:
            Z (torch.Tensor): Latent representation of shape (N, latentD).

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'pose_body' (torch.Tensor): Decoded pose in axis-angle
                    format (N, num_joints, 3).
                - 'pose_body_matrot' (torch.Tensor): Decoded pose as rotation
                    matrices (N, num_joints, 9).

        """
        bs = Z.shape[0]
        prec = self.decoder_net(Z)

        return {
            "pose_body": matrot2aa(prec.view(-1, 3, 3)).view(bs, -1, 3),
            "pose_body_matrot": prec.view(bs, -1, 9),
        }

    def forward(self, pose_body):
        """Forward pass with transparent pose remapping.

        If a PoseMapping is attached, pose_body may be in any source layout —
        it will be mapped to canonical format before encoding, and the decoded
        result will be mapped back to the original source layout.

        Extra joints (source joints not consumed by the canonical model) are
        preserved unchanged in 'pose_body_source'.

        Args:
            pose_body (torch.Tensor): Input body pose. Shape:
                - (N, num_source_joints * 3)  [flat], or
                - (N, num_source_joints, 3)   [structured].
                With IdentityMapper, num_source_joints == canonical num_joints.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'pose_body' (torch.Tensor): Decoded pose in canonical layout
                    (N, num_joints, 3).
                - 'pose_body_matrot' (torch.Tensor): Decoded pose as rotation
                    matrices in canonical layout (N, num_joints, 9).
                - 'pose_body_source' (torch.Tensor): Decoded pose remapped to
                    source layout (N, num_source_joints, 3). Extra joints are
                    reinserted unchanged. Identical to 'pose_body' when using
                    IdentityMapper.
                - 'poZ_body_mean' (torch.Tensor): Latent mean (N, latentD).
                - 'poZ_body_std' (torch.Tensor): Latent std (N, latentD).
                - 'q_z' (Normal): Full latent distribution.

        """
        # 1. Map source layout -> canonical
        mapped = self.pose_mapper.to_canonical(pose_body)
        canonical_pose = mapped["canonical_pose"]  # (N, J_canon, 3)
        extra_pose = mapped["extra_pose"]  # (N, E, 3) or None

        N = canonical_pose.shape[0]
        canonical_flat = canonical_pose.reshape(N, -1)

        # 2. Standard VAE forward in canonical space
        q_z = self.encode(canonical_flat)
        q_z_sample = q_z.rsample()
        decode_results = self.decode(q_z_sample)

        # 3. Map decoded pose back to source layout
        pose_body_source = self.pose_mapper.from_canonical(
            canonical_pose=decode_results["pose_body"],
            extra_pose=extra_pose,
            source_pose_original=pose_body,
        )

        decode_results.update({
            "pose_body_source": pose_body_source,
            "poZ_body_mean": q_z.mean,
            "poZ_body_std": q_z.scale,
            "q_z": q_z,
        })
        return decode_results

    def sample_poses(self, num_poses, temperature=1.0, seed=None):
        """Samples new poses from the latent space.

        Note: samples are returned in canonical layout. If you need source
        layout, use pose_mapper.from_canonical() on the result.

        Args:
            num_poses (int): Number of poses to generate.
            temperature (float): Scales the sampling std. Must be > 0.
            seed (int, optional): Random seed for reproducibility.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing generated poses
                in canonical layout.

        """
        np.random.seed(seed)

        assert temperature > 0.0, (
            f"Temperature must be positive and non-zero. Got {temperature}"
        )

        some_weight = next(self.parameters())
        dtype = some_weight.dtype
        device = some_weight.device

        self.eval()
        with torch.no_grad():
            Zgen = torch.tensor(
                np.random.normal(
                    0.0, 1.0 * temperature, size=(num_poses, self.latentD)
                ),
                dtype=dtype,
                device=device,
            )

        return self.decode(Zgen)

    def regularise_pose(self, full_pose: torch.Tensor) -> torch.Tensor:
        """Regularizes a pose by passing it through the VAE.

        The pose mapper is applied transparently — full_pose may be in either
        source or canonical layout depending on the attached mapper.

        Args:
            full_pose (torch.Tensor): A tensor of shape (batch_size, pose_dim)
                representing a full body pose.

        Returns:
            torch.Tensor: The regularized pose in the same layout as full_pose.

        """
        prepared_pose = remove_rotation_from_axis(full_pose, axis=2)

        # forward() returns pose_body_source, which is always in the input layout
        regularised_pose = self(prepared_pose)["pose_body_source"].reshape(
            full_pose.shape
        )

        output_pose = merge_global_orients_along_axis(
            full_pose, regularised_pose, axis=2
        )

        return output_pose


class VarenPoserTrainingExtension(VarenPoser):
    """Extended version of VarenPoser with additional training utilities.

    Includes functions for computing loss and constructing meshes.

    Attributes:
        body_model (VAREN): A pre-trained horse model used for pose
            reconstruction.

    """

    def __init__(
        self,
        varen_path,
        pose_mapper: Optional[BasePoseMapper] = None,
        **kwargs,
    ):
        """Initializes the extended VarenPoser class.

        Args:
            varen_path (str): Path to the pre-trained VAREN model.
            pose_mapper (BasePoseMapper, optional): See VarenPoser.__init__.
            **kwargs: Additional arguments.

        """
        super().__init__(pose_mapper=pose_mapper, **kwargs)
        self.body_model = BodyModel(varen_path)
        for param in self.body_model.parameters():
            param.requires_grad = False

    def construct_meshes(self, dorig, drec):
        """Constructs 3D meshes for original and reconstructed poses.

        Args:
            dorig (torch.Tensor): Original pose tensor of shape
                (N, num_joints * 3).
            drec (Dict[str, torch.Tensor]): Dictionary of reconstructed pose
                tensors. Uses 'pose_body_source' so that the mesh is built
                from the source-layout decoded pose.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Original and reconstructed mesh
                objects.

        """
        bs = dorig.shape[0]
        betas = torch.zeros(bs, 39).to(dorig.device).float()
        transl = torch.zeros(bs, 3).to(dorig.device).float()

        full_pose_in = dorig.float()
        # Use pose_body_source so mesh construction always uses canonical joints
        # even when a non-identity mapper is attached.
        full_pose_pred = drec["pose_body_source"].reshape(bs, -1).float()

        go_in = full_pose_in[:, :3]
        go_pred = full_pose_pred[:, :3]
        pose_in = full_pose_in[:, 3:]
        pose_pred = full_pose_pred[:, 3:]

        mesh_orig = self.body_model(
            global_orient=go_in, body_pose=pose_in, transl=transl, betas=betas
        )
        mesh_recon = self.body_model(
            global_orient=go_pred,
            body_pose=pose_pred,
            transl=transl,
            betas=betas,
        )

        return mesh_orig, mesh_recon

    def compute_loss(self, pose_gt, pose_pred):
        """Computes the total training loss.

        Loss components:
          - KL Divergence
          - L1 Mesh Reconstruction Loss
          - Geodesic Rotation Loss
          - Joint Position Loss

        Args:
            pose_gt (torch.Tensor): Ground-truth body pose (N, num_joints * 3).
            pose_pred (Dict[str, torch.Tensor]): Output dict from forward().

        Returns:
            Dict[str, Dict[str, torch.Tensor]]: Weighted and unweighted losses.

        """
        l1_loss = torch.nn.L1Loss(reduction="mean")
        geodesic_loss = geodesic_loss_R(reduction="mean")

        bs, latentD = pose_pred["poZ_body_mean"].shape
        device = pose_pred["poZ_body_mean"].device

        loss_kl_wt = 0.005
        loss_rec_wt = 4
        loss_matrot_wt = 2
        loss_jtr_wt = 2

        q_z = pose_pred["q_z"]

        mesh_orig, mesh_recon = self.construct_meshes(pose_gt, pose_pred)
        v2v = l1_loss(mesh_orig.vertices, mesh_recon.vertices)

        p_z = torch.distributions.normal.Normal(
            loc=torch.zeros((bs, latentD), device=device, requires_grad=False),
            scale=torch.ones(
                (bs, latentD), device=device, requires_grad=False
            ),
        )

        weighted_loss_dict = {
            "loss_kl": loss_kl_wt
            * torch.mean(
                torch.sum(
                    torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1]
                )
            ),
            "loss_mesh_rec": loss_rec_wt * v2v,
            "matrot": loss_matrot_wt
            * geodesic_loss(
                pose_pred["pose_body_matrot"].view(-1, 3, 3).double(),
                aa2matrot(pose_gt.view(-1, 3)),
            ),
            "jtr": loss_jtr_wt * l1_loss(mesh_orig.joints, mesh_recon.joints),
        }

        weighted_loss_dict["loss_total"] = torch.stack(
            list(weighted_loss_dict.values())
        ).sum()

        return {"weighted_loss": weighted_loss_dict, "unweighted_loss": {}}
