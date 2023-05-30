from typing import Tuple

import torch
from torch import nn

from .vae import VAE
from ..utils.loss import get_reconstruction_loss, get_kld_loss


class DIPVAE(VAE):
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


def compute_dip_vae_loss(
    input,
    dip_vae,
    dip_type,
    lambda_od,
    lambda_d,
    distribution="bernoulli",
    pixel_level=False,
    beta=1,
    *args,
    **kwargs,
) -> dict:
    output = dip_vae(input)
    decoded, mu, logvar, z = output

    reconstruction_loss = get_reconstruction_loss(decoded, input, distribution)
    kld_loss = get_kld_loss(mu, logvar)

    # cov(mu) = E((mu - E(mu))**2) = E(mu*mu^T) - E(mu)*E(mu)^T
    centered_mu = mu - mu.mean(0)
    cov_mu = (centered_mu.T @ centered_mu) / input.shape[0]

    target_cov = None
    if dip_type == "i":
        target_cov = cov_mu
    elif dip_type == "ii":
        # sigma = torch.sqrt(logvar.exp())
        # centered_sigma = sigma - sigma.mean(0)
        # cov_sigma = (centered_sigma.T @ centered_sigma) / input.shape[0]
        # cov_z = cov_sigma + cov_mu
        expection_sigma = torch.sqrt(logvar.exp()).mean(0)
        sigma = torch.diag_embed(expection_sigma, offset=0, dim1=-2, dim2=-1)
        cov_z = sigma + cov_mu
        # cov = torch.diag_embed(logvar.exp(), offset=0, dim1=-2, dim2=-1)
        # expectation_cov = torch.mean(cov, dim=0)
        # cov_z = expectation_cov + cov_mu
        target_cov = cov_z
    else:
        raise NotImplementedError(
            f"compute_dip_vae_loss dip_type = {dip_type}"
        )
    diag = torch.diagonal(target_cov, offset=0, dim1=-2, dim2=-1)
    off_diag = target_cov - torch.diag_embed(diag, offset=0, dim1=-2, dim2=-1)
    dip_off_diag_loss = torch.sum(off_diag**2)
    dip_diag_loss = torch.sum((diag - 1) ** 2)
    dip_loss = lambda_od * dip_off_diag_loss + lambda_d * dip_diag_loss
    if pixel_level:
        n_dims = torch.prod(torch.tensor(input.shape[1:]))
        loss = (reconstruction_loss + beta * kld_loss) / n_dims + dip_loss
    else:
        loss = reconstruction_loss + beta * kld_loss + dip_loss

    return dict(
        loss=loss,
        reconstruction_loss=reconstruction_loss,
        kld_loss=kld_loss,
        dip_off_diag_loss=dip_off_diag_loss,
        dip_diag_loss=dip_diag_loss,
        dip_loss=dip_loss,
    )
