from typing import List, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from .vae import VAE


class BetaVAE(VAE):
    def __init__(
        self,
        input_shape: Tuple[int],
        hidden_channels: List[int],
        latent_dim: int,
        beta: int,
    ) -> None:
        super().__init__(input_shape, hidden_channels, latent_dim)
        self.beta = beta

    def loss_function(
        self, input: Tensor, output: Union[Tensor, List[Tensor]], *args
    ) -> dict:
        decoded, mu, logvar, *_ = output
        batch_size = decoded.shape[0]
        reconstruction_loss = (
            F.mse_loss(input, decoded, reduction="sum") / batch_size
        )

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1),
            dim=0,
        )
        loss = reconstruction_loss + self.beta * kld_loss
        return dict(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            kld_loss=kld_loss,
        )
