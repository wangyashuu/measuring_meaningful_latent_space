from typing import Tuple

import torch
from torch import Tensor, nn

from .vae import VAE
from ..utils.loss import get_reconstruction_loss, get_kld_loss


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
    distribution="bernoulli",
    *args,
    **kwargs
) -> dict:
    output = beta_vae(input)
    decoded, mu, logvar, *_ = output
    reconstruction_loss = get_reconstruction_loss(decoded, input, distribution)
    kld_loss = get_kld_loss(mu, logvar)

    if c_max is not None:
        c_max = torch.tensor(c_max, device=input.device)
        c = torch.clamp(c_max * (step / n_c_steps), 0, c_max)
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
