from __future__ import annotations

import torch
from torch import nn
from varen import VAREN as BodyModel

from .pose_prior import QuadrupedPosePrior
from quadruped_poser.utils.losses import GeodesicRotationLoss
from quadruped_poser.utils.pose_transforms import aa2matrot


class QuadrupedPosePriorTrainer(QuadrupedPosePrior):
    """QuadrupedPosePrior extended with mesh construction and multi-term loss computation.

    Inherits the full VAE (encoder, latent space, decoder) from
    ``QuadrupedPosePrior`` and augments it with a frozen quadruped body model
    that is used to evaluate mesh-level reconstruction quality during training.

    Because the body model is only used as a differentiable renderer of body
    configurations, its parameters must not be updated during VAE training.
    All parameters of ``self.body_model`` are therefore frozen immediately
    after construction by setting ``requires_grad = False``.

    Args:
        body_model_path: Filesystem path to the pretrained quadruped body
            model checkpoint / configuration directory.
        **kwargs: Additional keyword arguments forwarded to
            ``QuadrupedPosePrior.__init__``.

    Attributes:
        body_model: Frozen quadruped body model instance used solely to
            produce vertices and joints for supervision.
    """

    def __init__(self, body_model_path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.body_model = BodyModel(body_model_path)
        for p in self.body_model.parameters():
            p.requires_grad = False

    def construct_meshes(
        self,
        pose_gt: torch.Tensor,
        pose_pred: dict[str, torch.Tensor],
    ) -> tuple:
        """Runs the body model for both the ground-truth and reconstructed poses.

        Creates zero-translation, zero-shape-blend body meshes for the
        original and predicted poses so that vertex and joint positions can be
        compared in 3D space.  Using the body model here ensures that
        reconstruction errors are measured in a geometrically meaningful space
        rather than directly in rotation-parameter space.

        Args:
            pose_gt: Ground-truth axis-angle pose tensor of shape
                ``(N, num_joints * 3)``.  The first three values encode the
                global orientation; the remainder encode body joints.
            pose_pred: Output dict from the VAE forward pass.  Must contain
                the key ``"pose_body"`` with shape ``(N, num_joints, 3)``.

        Returns:
            A tuple ``(mesh_orig, mesh_recon)`` where both elements are body
            model output objects with ``.vertices`` of shape
            ``(N, V, 3)`` and ``.joints`` of shape ``(N, J, 3)``.
        """
        bs = pose_gt.shape[0]
        betas  = torch.zeros(bs, 39, device=pose_gt.device).float()
        transl = torch.zeros(bs,  3, device=pose_gt.device).float()

        full_in   = pose_gt.float()
        full_pred = pose_pred['pose_body'].reshape(bs, -1).float()

        mesh_orig = self.body_model(
            global_orient=full_in[:, :3],
            body_pose=full_in[:, 3:],
            transl=transl,
            betas=betas,
        )
        mesh_recon = self.body_model(
            global_orient=full_pred[:, :3],
            body_pose=full_pred[:, 3:],
            transl=transl,
            betas=betas,
        )
        return mesh_orig, mesh_recon

    def compute_loss(
        self,
        pose_gt: torch.Tensor,
        pose_pred: dict[str, torch.Tensor],
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Computes the total weighted VAE training loss from four components.

        The overall objective is:

        .. math::

            \\mathcal{L} = w_{\\text{KL}} \\cdot D_{\\text{KL}} +
                           w_{\\text{mesh}} \\cdot \\mathcal{L}_{\\text{mesh}} +
                           w_{\\text{rot}} \\cdot \\mathcal{L}_{\\text{rot}} +
                           w_{\\text{jtr}} \\cdot \\mathcal{L}_{\\text{jtr}}

        **KL divergence** (weight 0.005):

        .. math::

            D_{\\text{KL}} = \\mathbb{E}_x \\left[
                \\sum_{d=1}^{D} \\text{KL}\\!\\left(
                    \\mathcal{N}(\\mu_d, \\sigma_d^2) \\,\\|\\, \\mathcal{N}(0, 1)
                \\right)
            \\right]

        where the closed-form per-dimension KL is
        :math:`\\frac{1}{2}(\\mu_d^2 + \\sigma_d^2 - \\log \\sigma_d^2 - 1)`.
        Summing over the latent dimension ``D`` and averaging over the batch
        regularises the posterior towards the unit Gaussian prior.

        **Vertex L1 loss** (weight 4):

        .. math::

            \\mathcal{L}_{\\text{mesh}} = \\frac{1}{NV}
                \\sum_{n,v} \\| \\hat{V}_{n,v} - V_{n,v} \\|_1

        Mean absolute vertex displacement between the original and
        reconstructed meshes in 3D space, encouraging anatomically plausible
        surface reconstruction.

        **Geodesic rotation loss** (weight 2):

        .. math::

            \\mathcal{L}_{\\text{rot}} = \\frac{1}{NJ}
                \\sum_{n,j} \\arccos\\!\\left(
                    \\frac{\\text{tr}(R_{n,j}^{\\top} \\hat{R}_{n,j}) - 1}{2}
                \\right)

        The geodesic distance between ground-truth and predicted rotation
        matrices measures the shortest angular path on SO(3), making it
        invariant to the choice of rotation representation.

        **Joint-trajectory L1 loss** (weight 2):

        .. math::

            \\mathcal{L}_{\\text{jtr}} = \\frac{1}{NJ}
                \\sum_{n,j} \\| \\hat{j}_{n,j} - j_{n,j} \\|_1

        Mean absolute joint-position error in 3D space, computed from the
        body model's kinematic skeleton.

        Args:
            pose_gt: Ground-truth axis-angle pose tensor of shape
                ``(N, num_joints * 3)``.
            pose_pred: Output dict from the VAE forward pass.  Must contain
                ``"poZ_body_mean"``, ``"pose_body_matrot"``, and ``"q_z"``.

        Returns:
            A dict with two keys:

            * ``"weighted_loss"`` — dict mapping loss name to its weighted
              scalar tensor.  Contains ``"loss_kl"``, ``"loss_mesh_rec"``,
              ``"matrot"``, ``"jtr"``, and ``"loss_total"`` (their sum).
            * ``"unweighted_loss"`` — empty dict, reserved for future use.
        """
        l1_loss       = nn.L1Loss(reduction='mean')
        geodesic_loss = GeodesicRotationLoss(reduction='mean')

        bs, latentD = pose_pred['poZ_body_mean'].shape
        device      = pose_pred['poZ_body_mean'].device

        q_z = pose_pred['q_z']
        p_z = torch.distributions.Normal(
            torch.zeros((bs, latentD), device=device),
            torch.ones((bs, latentD), device=device),
        )

        mesh_orig, mesh_recon = self.construct_meshes(pose_gt, pose_pred)

        weighted = {
            'loss_kl': 0.005 * torch.mean(
                torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1])
            ),
            'loss_mesh_rec': 4 * l1_loss(mesh_orig.vertices, mesh_recon.vertices),
            'matrot': 2 * geodesic_loss(
                pose_pred['pose_body_matrot'].view(-1, 3, 3).double(),
                aa2matrot(pose_gt.view(-1, 3)),
            ),
            'jtr': 2 * l1_loss(mesh_orig.joints, mesh_recon.joints),
        }
        weighted['loss_total'] = torch.stack(list(weighted.values())).sum()

        return {'weighted_loss': weighted, 'unweighted_loss': {}}
