from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BatchFlatten(nn.Module):
    """Flattens all dimensions of a tensor except the batch dimension.

    Reshapes an input tensor of shape (N, d1, d2, ...) into (N, d1 * d2 * ...).
    This is a convenience wrapper around ``Tensor.view`` that always preserves
    the leading batch axis, making it safe to use inside ``nn.Sequential``
    pipelines where the batch size is not known at construction time.

    Example:
        >>> m = BatchFlatten()
        >>> x = torch.randn(4, 3, 8)
        >>> m(x).shape
        torch.Size([4, 24])
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten all non-batch dimensions.

        Args:
            x: Input tensor of shape (N, *).

        Returns:
            Tensor of shape (N, product-of-remaining-dims).
        """
        return x.view(x.shape[0], -1)


class View(nn.Module):
    """Reshapes a tensor to an arbitrary fixed shape inside ``nn.Sequential``.

    Unlike ``BatchFlatten``, the target shape must be fully specified at
    construction time (including the batch dimension). This makes the module
    suitable for decoder pipelines where the spatial layout is known in advance.

    Args:
        *shape: Target shape passed directly to ``Tensor.view``.

    Example:
        >>> m = View(2, 3, 4)
        >>> x = torch.randn(24)
        >>> m(x).shape
        torch.Size([2, 3, 4])
    """

    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape the input tensor.

        Args:
            x: Input tensor whose total number of elements must equal the
               product of ``self.shape``.

        Returns:
            Tensor reshaped to ``self.shape``.
        """
        return x.view(self.shape)


class OrthoRotDecoder(nn.Module):
    """Converts a 6D continuous rotation representation to a 3x3 rotation matrix.

    Implements the Gram-Schmidt-based orthogonalisation procedure introduced by
    Zhou et al. (2019), *"On the Continuity of Rotation Representations in
    Neural Networks"* (CVPR 2019).  The 6D input encodes two 3-vectors
    ``(a1, a2)`` that are treated as the first two columns of the target
    rotation matrix.  The orthonormal basis ``(b1, b2, b3)`` is recovered as
    follows:

    .. math::

        b_1 &= \\frac{a_1}{\\|a_1\\|}

        b_2 &= \\frac{a_2 - (b_1 \\cdot a_2)\\, b_1}{\\|a_2 - (b_1 \\cdot a_2)\\, b_1\\|}

        b_3 &= b_1 \\times b_2

    The resulting matrix :math:`R = [b_1 \\mid b_2 \\mid b_3]` is guaranteed to
    be a valid rotation matrix (``det(R) = +1``) for any non-degenerate input,
    which makes this representation well-suited for gradient-based optimisation.

    Input shape: ``(..., 6)`` — two 3D column vectors concatenated per joint.
    Output shape: ``(..., 3, 3)``.

    Note:
        This module is intentionally stateless (no learnable parameters).
        It is typically placed at the end of a decoder ``nn.Sequential``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gram-Schmidt orthogonalisation to produce rotation matrices.

        Args:
            x: Tensor of shape (N, 6) or any shape whose last dimension is 6,
               containing two 3D vectors per sample.

        Returns:
            Tensor of shape (N, 3, 3) containing valid rotation matrices.
        """
        x = x.view(-1, 3, 2)
        b1 = F.normalize(x[:, :, 0], dim=1)
        dot = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(x[:, :, 1] - dot * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=-1)


class LatentDistHead(nn.Module):
    """Linear head that maps a feature vector to Normal distribution parameters.

    Projects a hidden representation onto the parameters of a diagonal
    multivariate Normal distribution :math:`q(z | x) = \\mathcal{N}(\\mu, \\sigma^2 I)`,
    which serves as the approximate posterior in a VAE.

    Two independent linear layers produce:

    * **mean** ``mu``: the centre of the posterior distribution.
    * **log-variance** ``logvar``: passed through ``softplus`` to obtain the
      strictly positive scale parameter ``sigma``.  Using ``softplus`` (rather
      than ``exp``) avoids numerical overflow when logvar becomes large.

    The ``forward`` method returns a ``torch.distributions.Normal`` object so
    that callers can use ``.rsample()`` for the reparameterisation trick and
    ``kl_divergence`` helpers directly.

    Args:
        in_features: Dimensionality of the input feature vector.
        latent_dim: Dimensionality of the latent space ``z``.

    Example:
        >>> head = LatentDistHead(512, 32)
        >>> h = torch.randn(8, 512)
        >>> dist = head(h)
        >>> z = dist.rsample()   # reparameterised sample, shape (8, 32)
    """

    def __init__(self, in_features: int, latent_dim: int) -> None:
        super().__init__()
        self.mu = nn.Linear(in_features, latent_dim)
        self.logvar = nn.Linear(in_features, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.distributions.Normal:
        """Compute the posterior Normal distribution from a feature vector.

        Args:
            x: Feature tensor of shape (N, in_features).

        Returns:
            A ``torch.distributions.Normal`` with ``mean`` of shape
            (N, latent_dim) and ``scale`` of shape (N, latent_dim).
            The scale is obtained via ``softplus(logvar(x))`` to ensure
            strict positivity.
        """
        return torch.distributions.Normal(self.mu(x), F.softplus(self.logvar(x)))
