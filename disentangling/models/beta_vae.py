from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .vae import VAE


class BetaVAE(VAE):
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


def compute_beta_vae_loss(
    input: Tensor,
    beta_vae,
    beta=1,
    c_max=None,
    n_c_steps=None,
    step=None,
    *args,
    **kwargs
) -> dict:
    output = beta_vae(input)
    decoded, mu, logvar, *_ = output
    batch_size = decoded.shape[0]
    reconstruction_loss = (
        F.mse_loss(input, decoded, reduction="sum") / batch_size
    )
    kld_loss = (
        -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    ) / batch_size

    if c_max is not None:
        c = torch.clamp((step / n_c_steps) * c_max, 0, c_max)
        loss = reconstruction_loss + beta * (kld_loss - c).abs()
        return dict(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            kld_loss=kld_loss,
            c=c,
        )

    loss = reconstruction_loss + beta * kld_loss
    return dict(
        loss=loss,
        reconstruction_loss=reconstruction_loss,
        kld_loss=kld_loss,
    )
