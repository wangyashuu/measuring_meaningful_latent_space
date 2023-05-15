from typing import Tuple

import torch
from torch import nn

from .vae import VAE
from ..utils.loss import get_reconstruction_loss, get_kld_loss

## https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder/blob/master/mmd_vae.ipynb


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
        -torch.mean((z1[:, None, :] - z2[None, :, :]) ** 2, dim=-1)
        / (1.0 * n_latent_dim * sigma**2)
    )
    return k


def init_weights(m):
    if (
        isinstance(m, nn.Linear)
        or isinstance(m, nn.ConvTranspose2d)
        or isinstance(m, nn.Conv2d)
    ):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


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
        self.apply(init_weights)


def compute_info_vae_loss(
    input,
    info_vae,
    alpha,
    lambd,
    kernel_type="rbf",
    kernel_sigma=1,
    distribution="bernoulli",
    *args,
    **kwargs
) -> dict:
    # TODO: refactor
    # kld_loss_factor = 1 - alpha
    # mmd_loss_factor = gamma - (1-alpha)

    output = info_vae(input)
    decoded, mu, logvar, z = output

    reconstruction_loss = get_reconstruction_loss(decoded, input, distribution)
    kld_loss = get_kld_loss(mu, logvar)

    z_prior = torch.randn_like(z, device=z.device)
    z_posterior = z
    compute_kernel = rbf_kernel if kernel_type == "rbf" else imq_kernel
    z_prior_kernel = compute_kernel(z_prior, z_prior, sigma=kernel_sigma)
    z_posterior_kernel = compute_kernel(
        z_posterior, z_posterior, sigma=kernel_sigma
    )
    cross_kernel = compute_kernel(z_prior, z_posterior, sigma=kernel_sigma)
    mmd_loss = (
        z_posterior_kernel.mean()
        + z_prior_kernel.mean()
        - 2 * cross_kernel.mean()
    )
    n_dims = torch.prod(torch.tensor(input.shape[1:]))
    loss = (
        (reconstruction_loss / n_dims)
        + (1 - alpha) * (kld_loss / n_dims)
        + (alpha + lambd - 1) * mmd_loss
    )

    return dict(
        loss=loss,
        reconstruction_loss=reconstruction_loss,
        kld_loss=kld_loss,
        mmd_loss=mmd_loss,
    )
