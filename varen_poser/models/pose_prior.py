import torch
import numpy as np
from torch import nn

from .components import BatchFlatten, OrthoRotDecoder, LatentDistHead
from varen_poser.utils.rotation_tools import (
    matrot2aa,
    remove_rotation_from_axis,
    merge_global_orients_along_axis,
)

_NUM_JOINTS = 38   # 37 body joints + global orient
_LATENT_DIM = 32
_HIDDEN_DIM = 512


class QuadrupedPosePrior(nn.Module):
    """VAE-based pose prior for quadruped body models.

    Learns a compact latent representation of plausible body poses via a
    variational autoencoder. The encoder maps axis-angle poses to a Gaussian
    latent space; the decoder maps latent codes back to rotation matrices and
    axis-angle poses.

    Attributes:
        num_joints: Number of joints (including global orientation).
        latentD: Dimensionality of the latent space.
        encoder_net: Sequential encoder network.
        decoder_net: Sequential decoder network.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.num_joints = _NUM_JOINTS
        self.latentD = _LATENT_DIM
        n_features = self.num_joints * 3

        self.encoder_net = nn.Sequential(
            BatchFlatten(),
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, _HIDDEN_DIM),
            nn.LeakyReLU(),
            nn.BatchNorm1d(_HIDDEN_DIM),
            nn.Dropout(0.1),
            nn.Linear(_HIDDEN_DIM, _HIDDEN_DIM),
            nn.Linear(_HIDDEN_DIM, _HIDDEN_DIM),
            LatentDistHead(_HIDDEN_DIM, self.latentD),
        )

        self.decoder_net = nn.Sequential(
            nn.Linear(self.latentD, _HIDDEN_DIM),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(_HIDDEN_DIM, _HIDDEN_DIM),
            nn.LeakyReLU(),
            nn.Linear(_HIDDEN_DIM, self.num_joints * 6),
            OrthoRotDecoder(),
        )

    def encode(self, pose_body: torch.Tensor) -> torch.distributions.Normal:
        """Encodes a pose into a latent Normal distribution.

        Args:
            pose_body: Tensor of shape (N, num_joints * 3).

        Returns:
            Normal distribution over the latent space.
        """
        return self.encoder_net(pose_body)

    def decode(self, z: torch.Tensor) -> dict:
        """Decodes a latent vector into pose representations.

        Args:
            z: Latent tensor of shape (N, latentD).

        Returns:
            Dict with keys:
              - ``pose_body``: axis-angle pose, shape (N, num_joints, 3).
              - ``pose_body_matrot``: rotation matrices, shape (N, num_joints, 9).
        """
        bs = z.shape[0]
        rot_mats = self.decoder_net(z)  # (N * num_joints, 3, 3)
        return {
            'pose_body': matrot2aa(rot_mats.view(-1, 3, 3)).view(bs, -1, 3),
            'pose_body_matrot': rot_mats.view(bs, -1, 9),
        }

    def forward(self, pose_body: torch.Tensor) -> dict:
        """Encodes then decodes a pose (training forward pass).

        Args:
            pose_body: Tensor of shape (N, num_joints * 3).

        Returns:
            Decode dict plus ``poZ_body_mean``, ``poZ_body_std``, ``q_z``.
        """
        q_z = self.encode(pose_body)
        out = self.decode(q_z.rsample())
        out.update({
            'poZ_body_mean': q_z.mean,
            'poZ_body_std': q_z.scale,
            'q_z': q_z,
        })
        return out

    def sample_poses(self, num_poses: int, temperature: float = 1.0, seed: int = None) -> dict:
        """Samples novel poses by drawing from the prior.

        Args:
            num_poses: Number of poses to generate.
            temperature: Scale of the prior samples; higher values give more
                diverse, less typical poses.
            seed: Optional random seed for reproducibility.

        Returns:
            Decode dict containing the sampled poses.
        """
        assert temperature > 0.0, (
            f"Temperature must be positive and non-zero. Got {temperature}"
        )
        np.random.seed(seed)

        param = next(self.parameters())
        self.eval()
        with torch.no_grad():
            z = torch.tensor(
                np.random.normal(0.0, temperature, size=(num_poses, self.latentD)),
                dtype=param.dtype,
                device=param.device,
            )
        return self.decode(z)

    def regularise_pose(self, full_pose: torch.Tensor) -> torch.Tensor:
        """Projects a pose onto the learned prior while preserving z-rotation.

        Strips the global z-rotation, passes the pose through the VAE, then
        restores the original z-rotation.

        Args:
            full_pose: Tensor of shape (N, num_joints * 3).

        Returns:
            Regularised pose tensor with the same shape as the input.
        """
        stripped = remove_rotation_from_axis(full_pose, axis=2)
        regularised = self(stripped)['pose_body'].reshape(full_pose.shape)
        return merge_global_orients_along_axis(full_pose, regularised, axis=2)
