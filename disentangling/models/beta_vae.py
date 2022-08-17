from base64 import encode
from typing import List, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .vae import VAE


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    Reparameterization trick to sample from N(mu, var) from N(0,1).
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


class BetaVAE(VAE):
    def __init__(
        self,
        in_channels: int,
        in_size: Tuple[int],
        hidden_channels: List[int],
        latent_dim: int,
        beta: int,
    ) -> None:
        super().__init__(in_channels, in_size, hidden_channels, latent_dim)
        self.beta = beta

    def loss_function(self, input, output) -> dict:
        decoded, mu, logvar = output
        reconstruction_loss = (
            F.mse_loss(input, decoded, reduction="none").sum(-1).mean()
        )

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1),
            dim=0,
        )
        loss = reconstruction_loss + self.beta * kld_loss
        return loss
