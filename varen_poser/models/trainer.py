import torch
from torch import nn
from varen import VAREN

from .pose_prior import QuadrupedPosePrior
from varen_poser.utils.losses import GeodesicRotationLoss
from varen_poser.utils.pose_transforms import aa2matrot


class QuadrupedPosePriorTrainer(QuadrupedPosePrior):
    """QuadrupedPosePrior extended with mesh construction and loss computation.

    The body model is loaded in inference-only mode; its parameters are frozen
    and excluded from optimisation.

    Attributes:
        body_model: Frozen quadruped body model used for vertex/joint supervision.
    """

    def __init__(self, varen_path: str, **kwargs):
        super().__init__(**kwargs)
        self.body_model = VAREN(varen_path)
        for p in self.body_model.parameters():
            p.requires_grad = False

    def construct_meshes(self, pose_gt: torch.Tensor, pose_pred: dict):
        """Runs the body model for the ground-truth and predicted poses.

        Args:
            pose_gt: Ground-truth pose tensor of shape (N, num_joints * 3).
            pose_pred: Output dict from the VAE forward pass.

        Returns:
            Tuple of (original mesh, reconstructed mesh).
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

    def compute_loss(self, pose_gt: torch.Tensor, pose_pred: dict) -> dict:
        """Computes the total VAE training loss.

        Components:
        - KL divergence between the posterior and the standard normal prior.
        - L1 vertex-to-vertex mesh reconstruction loss.
        - Geodesic rotation loss on the rotation matrices.
        - L1 joint-trajectory loss.

        Args:
            pose_gt: Ground-truth pose tensor of shape (N, num_joints * 3).
            pose_pred: Output dict from the VAE forward pass.

        Returns:
            Dict with keys ``weighted_loss`` and ``unweighted_loss``.
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
