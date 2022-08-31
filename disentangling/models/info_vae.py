from distutils.util import rfc822_escape
from typing import List, Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .vae import VAE


def imq_kernel(z1, z2, sigma, scales=[0.1, 0.2, 0.5, 1.0, 2.0, 5, 10.0]):
    """
    TODO: inverse multiquadric kernel
    TODO: what is scale
    https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder
    """
    n_latent_dim = z1.shape[1]
    Cbase = 2.0 * n_latent_dim * sigma**2
    k = 0
    for scale in scales:
        C = scale * Cbase
        k += C / (
            C + torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2
        )

    return k


def rbf_kernel(z1, z2, n_latent_dim, sigma):
    """
    TODO: Radial basis function kernel
    https://en.wikipedia.org/wiki/Radial_basis_function_kernel
    """
    n_latent_dim = z1.shape[1]
    k = torch.exp(
        -torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2
        / (2.0 * n_latent_dim * sigma**2)
    )
    return k


class InfoVAE(VAE):
    "TODO: https://github.com/fducau/infoVAE/issues/2"

    def __init__(
        self,
        input_shape: Tuple[int],
        hidden_channels: List[int],
        latent_dim: int,
        kernel_type: str = "rbf",
        kernel_sigma: float = 1.0,
    ) -> None:
        super().__init__(input_shape, hidden_channels, latent_dim)
        self.kernel_type = kernel_type
        self.kernel_sigma = kernel_sigma

    def loss_function(
        self, input: Tensor, output: Union[Tensor, List[Tensor]], **kwargs
    ) -> dict:
        decoded, mu, logvar, z = output
        batch_size = decoded.shape[0]
        reconstruction_loss = (
            F.mse_loss(input, decoded, reduction="sum") / batch_size
        )

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1),
            dim=0,
        )

        N = batch_size

        z_posterior = z
        z_prior = torch.randn_like(z, device=z.device)

        compute_kernel = (
            rbf_kernel if self.kernel_type == "rbf" else imq_kernel
        )

        z_posterior_kernel = compute_kernel(
            z_posterior, z_posterior, self.kernel_sigma
        )
        z_prior_kernel = compute_kernel(z_prior, z_prior, self.kernel_sigma)
        cross_kernel = compute_kernel(z_prior, z_posterior, self.kernel_sigma)

        mmd_loss = (
            (z_posterior_kernel - z_posterior_kernel.diag().diag()).sum()
            / ((N - 1) * N)
            + (z_prior_kernel - z_prior_kernel.diag().diag()).sum()
            / ((N - 1) * N)
            + -2 * cross_kernel.sum() / (N**2)
        )

        loss = (
            reconstruction_loss
            + (1 - self.alpha) * kld_loss
            + (self.alpha + self.gamma - 1) * mmd_loss
        )
        return loss
