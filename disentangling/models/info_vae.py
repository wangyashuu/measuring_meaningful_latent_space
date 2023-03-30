from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .vae import VAE


def imq_kernel(z1, z2, sigma, scales=[1.0]):
    """
    TODO: inverse multiquadric kernel
    TODO: what is scale [0.1, 0.2, 0.5, 1.0, 2.0, 5, 10.0]
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


def rbf_kernel(z1, z2, sigma):
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
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encoder_output_shape: Tuple,
        decoder_input_shape: Tuple,
        latent_dim: int,
    ) -> None:
        super().__init__(
            encoder,
            decoder,
            encoder_output_shape,
            decoder_input_shape,
            latent_dim,
        )


def compute_info_vae_loss(
    input,
    info_vae,
    alpha,
    lambd,
    kernel_type="rbf",
    kernel_sigma=1,
    *args,
    **kwargs
) -> dict:
    # TODO: refactor
    # kld_loss_factor = 1 - alpha
    # mmd_loss_factor = gamma - (1-alpha)

    output = info_vae(input)
    decoded, mu, logvar, z = output
    batch_size = decoded.shape[0]

    reconstruction_loss = (
        F.mse_loss(input, decoded, reduction="sum") / batch_size
    )
    kld_loss = (
        -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    ) / batch_size

    z_posterior = z
    z_prior = torch.randn_like(z, device=z.device)

    compute_kernel = rbf_kernel if kernel_type == "rbf" else imq_kernel
    z_posterior_kernel = compute_kernel(
        z_posterior, z_posterior, sigma=kernel_sigma
    )
    z_prior_kernel = compute_kernel(z_prior, z_prior, sigma=kernel_sigma)
    cross_kernel = compute_kernel(z_prior, z_posterior, sigma=kernel_sigma)

    mmd_z_posterior = (  # z_posterior_kernel.mean()
        z_posterior_kernel - z_posterior_kernel.diag().diag()
    ).sum() / ((batch_size - 1) * batch_size)
    mmd_z_prior = (  # z_prior_kernel.mean()
        z_prior_kernel - z_prior_kernel.diag().diag()
    ).sum() / ((batch_size - 1) * batch_size)
    mmd_cross = cross_kernel.mean()
    mmd_loss = mmd_z_posterior + mmd_z_prior - 2 * mmd_cross
    loss = (
        reconstruction_loss
        + (1 - alpha) * kld_loss
        + (alpha + lambd - 1) * mmd_loss
    )
    return dict(loss=loss, kld_loss=kld_loss, mmd_loss=mmd_loss)
