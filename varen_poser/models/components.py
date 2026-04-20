import torch
from torch import nn
from torch.nn import functional as F


class BatchFlatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class OrthoRotDecoder(nn.Module):
    """Converts a 6D continuous rotation representation to a 3x3 rotation matrix.

    Uses the Gram-Schmidt-like orthogonalisation from Zhou et al. (2019).
    Input shape: (..., 6) — two 3D column vectors per rotation.
    Output shape: (..., 3, 3).
    """

    def forward(self, x):
        x = x.view(-1, 3, 2)
        b1 = F.normalize(x[:, :, 0], dim=1)
        dot = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(x[:, :, 1] - dot * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=-1)


class LatentDistHead(nn.Module):
    """Linear head that maps a feature vector to a Normal distribution.

    Args:
        in_features: Size of the input feature vector.
        latent_dim: Dimensionality of the latent space.
    """

    def __init__(self, in_features: int, latent_dim: int):
        super().__init__()
        self.mu = nn.Linear(in_features, latent_dim)
        self.logvar = nn.Linear(in_features, latent_dim)

    def forward(self, x):
        return torch.distributions.Normal(self.mu(x), F.softplus(self.logvar(x)))
