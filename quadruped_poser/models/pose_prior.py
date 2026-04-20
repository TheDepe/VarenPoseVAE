from __future__ import annotations

import torch
import numpy as np
from torch import nn

from .components import BatchFlatten, OrthoRotDecoder, LatentDistHead
from quadruped_poser.utils.pose_transforms import (
    matrot2aa,
    remove_rotation_from_axis,
    merge_global_orients_along_axis,
)

_NUM_JOINTS = 38   # 37 body joints + global orient
_LATENT_DIM = 32
_HIDDEN_DIM = 512


class QuadrupedPosePrior(nn.Module):
    """VAE-based pose prior for quadruped body models.

    Learns a compact, low-dimensional representation of plausible body poses
    for quadruped animals via a variational autoencoder (VAE).  The overall
    architecture follows the standard VAE pipeline:

    1. **Encoder** — a two-hidden-layer MLP with BatchNorm and Dropout maps a
       flattened axis-angle pose vector ``(N, num_joints * 3)`` to the
       parameters of a diagonal Gaussian posterior
       ``q(z | x) = N(mu, sigma^2 I)`` in ``R^{latentD}``.

    2. **Latent sampling** — a sample ``z`` is drawn from the posterior using
       the reparameterisation trick (``rsample``), enabling gradients to flow
       through the stochastic node during training.

    3. **Decoder** — a two-hidden-layer MLP maps ``z`` back to a 6D continuous
       rotation representation ``(N * num_joints, 6)``.  A stateless
       ``OrthoRotDecoder`` layer converts these 6D vectors to valid 3×3
       rotation matrices via Gram-Schmidt orthogonalisation.

    4. **Pose output** — the rotation matrices are converted back to axis-angle
       form via ``matrot2aa``, yielding a pose tensor of shape
       ``(N, num_joints, 3)``.

    The model is trained with a KL-divergence regularisation term that
    encourages the posterior to stay close to the unit Gaussian prior
    ``p(z) = N(0, I)``, alongside reconstruction losses in rotation-matrix and
    vertex space.

    Attributes:
        num_joints: Number of joints modelled, including global orientation
            (default 38 = 37 body joints + 1 global).
        latentD: Dimensionality of the latent space (default 32).
        encoder_net: Sequential encoder network ending with a
            ``LatentDistHead`` that returns a ``torch.distributions.Normal``.
        decoder_net: Sequential decoder network ending with an
            ``OrthoRotDecoder`` that returns rotation matrices.
    """

    def __init__(self, **kwargs) -> None:
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

        Passes the input pose through the encoder MLP, which produces the mean
        and scale of the approximate posterior ``q(z | x)``.  The returned
        distribution object can be used directly for sampling (``rsample`` for
        the reparameterisation trick) and for computing KL divergence against
        the prior.

        Args:
            pose_body: Axis-angle pose tensor of shape ``(N, num_joints * 3)``
                where each triplet encodes one joint's rotation.

        Returns:
            A ``torch.distributions.Normal`` with ``mean`` and ``scale`` each
            of shape ``(N, latentD)``, representing the per-sample posterior.
        """
        return self.encoder_net(pose_body)

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decodes a latent vector into pose representations.

        Maps a point in latent space back to a full body pose.  The decoder
        first produces a 6D continuous rotation representation for every joint,
        which is then converted to a valid 3×3 rotation matrix via
        Gram-Schmidt orthogonalisation (``OrthoRotDecoder``).  Finally, the
        rotation matrices are converted to axis-angle form for compatibility
        with downstream body models.

        Args:
            z: Latent tensor of shape ``(N, latentD)``.

        Returns:
            A dict with two entries:

            * ``"pose_body"`` — axis-angle pose, shape ``(N, num_joints, 3)``.
            * ``"pose_body_matrot"`` — flattened rotation matrices, shape
              ``(N, num_joints, 9)``.
        """
        bs = z.shape[0]
        rot_mats = self.decoder_net(z)  # (N * num_joints, 3, 3)
        return {
            'pose_body': matrot2aa(rot_mats.view(-1, 3, 3)).view(bs, -1, 3),
            'pose_body_matrot': rot_mats.view(bs, -1, 9),
        }

    def forward(self, pose_body: torch.Tensor) -> dict[str, torch.Tensor]:
        """Runs the full VAE encode-decode pass used during training.

        Encodes the input pose to a posterior distribution, draws a
        reparameterised sample with ``rsample`` (enabling gradient flow
        through the stochastic node), and decodes that sample back to pose
        space.  The returned dict contains everything needed to compute the
        VAE loss (reconstruction terms + KL divergence).

        Args:
            pose_body: Axis-angle pose tensor of shape ``(N, num_joints * 3)``.

        Returns:
            The decode dict (see ``decode``) extended with three additional
            entries:

            * ``"poZ_body_mean"`` — posterior mean, shape ``(N, latentD)``.
            * ``"poZ_body_std"`` — posterior scale, shape ``(N, latentD)``.
            * ``"q_z"`` — the full ``torch.distributions.Normal`` posterior,
              used to compute the KL term in the training loss.
        """
        q_z = self.encode(pose_body)
        out = self.decode(q_z.rsample())
        out.update({
            'poZ_body_mean': q_z.mean,
            'poZ_body_std': q_z.scale,
            'q_z': q_z,
        })
        return out

    def sample_poses(
        self,
        num_poses: int,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Samples novel poses by drawing directly from the prior.

        Generates poses by sampling ``z ~ N(0, temperature^2 * I)`` and
        decoding each sample.  Higher temperatures explore regions of latent
        space further from the origin, yielding more diverse but potentially
        less typical poses.  The model is temporarily placed in eval mode and
        ``torch.no_grad()`` is used to avoid unnecessary gradient computation.

        Args:
            num_poses: Number of poses to generate.
            temperature: Standard deviation of the Gaussian prior samples.
                Must be strictly positive.  Defaults to ``1.0`` (standard
                normal prior).
            seed: Optional integer seed passed to ``numpy.random.seed`` for
                reproducible sampling.

        Returns:
            The decode dict (see ``decode``) containing the sampled poses.

        Raises:
            AssertionError: If ``temperature`` is not strictly positive.
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

        Strips the global z-axis rotation from the input pose so that the VAE
        is not distracted by global orientation, passes the stripped pose
        through the full VAE forward pass, and then restores the original
        z-rotation in the reconstructed output.  This procedure ensures that
        regularisation acts only on the body configuration, not on how the
        animal is oriented in the world.

        Args:
            full_pose: Axis-angle pose tensor of shape ``(N, num_joints * 3)``,
                including the global orientation as the first three values.

        Returns:
            Regularised pose tensor of the same shape as ``full_pose``, with
            body joints pulled towards the learned prior distribution and the
            original global z-rotation preserved.
        """
        stripped = remove_rotation_from_axis(full_pose, axis=2)
        regularised = self(stripped)['pose_body'].reshape(full_pose.shape)
        return merge_global_orients_along_axis(full_pose, regularised, axis=2)
