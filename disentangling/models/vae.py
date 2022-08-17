from base64 import encode
from typing import List, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .base_ae import BaseAE


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    Reparameterization trick to sample from N(mu, var) from N(0,1).
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


class VAE(BaseAE):
    def __init__(
        self,
        input_shape: Tuple[int],
        hidden_channels: List[int],
        latent_dim: int
    ) -> None:
        super().__init__(
            encoder_input_shape=input_shape,
            encoder_output_dim=latent_dim * 2,
            decoder_input_dim=latent_dim,
            hidden_channels=hidden_channels,
        )

    def forward(self, input) -> List[Tensor]:
        encoded = self.encoder(input)
        latent_dim = encoded.shape[1] // 2
        mu, logvar = encoded[:,:latent_dim], encoded[:,latent_dim:]
        z = reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return decoded, mu, logvar

    def loss_function(self, input, output) -> dict:
        decoded, mu, logvar = output
        reconstruction_loss = (
            F.mse_loss(input, decoded, reduction="none").sum(-1).mean()
        )

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1),
            dim=0,
        )
        loss = reconstruction_loss + kld_loss
        return loss
