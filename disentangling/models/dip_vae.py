from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .vae import VAE


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
    input, dip_vae, dip_type, lambda_od, lambda_d, *args, **kwargs
) -> dict:
    output = dip_vae(input)
    decoded, mu, logvar, z = output
    batch_size = decoded.shape[0]
    reconstruction_loss = (
        F.mse_loss(input, decoded, reduction="sum") / batch_size
    )
    kld_loss = (
        -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    ) / batch_size
    expectation_mu_mu_t = (mu[:, :, None] @ mu[:, None, :]).mean(0)
    expectation_mu = mu.mean(0)
    # TODO: how to compute covariance
    cov_mu = (
        expectation_mu_mu_t - expectation_mu[None, :] @ expectation_mu[:, None]
    )

    target_cov = None
    if dip_type == "i":
        target_cov = cov_mu
    elif dip_type == "ii":
        cov = torch.diag_embed(logvar.exp(), offset=0, dim1=-2, dim2=-1)
        expectation_cov = torch.mean(cov, dim=0)
        cov_z = expectation_cov + cov_mu
        target_cov = cov_z
    else:
        raise NotImplementedError(f"dip_type = {dip_type}")
    diag = torch.diagonal(target_cov, offset=0, dim1=-2, dim2=-1)
    off_diag = diag - torch.diag_embed(diag, offset=0, dim1=-2, dim2=-1)
    dip_loss = lambda_od * torch.sum(off_diag**2) + lambda_d * torch.sum(
        (diag - 1) ** 2
    )

    loss = reconstruction_loss + kld_loss + dip_loss
    return dict(
        loss=loss,
        reconstruction_loss=reconstruction_loss,
        kld_loss=kld_loss,
        dip_loss=dip_loss,
    )
