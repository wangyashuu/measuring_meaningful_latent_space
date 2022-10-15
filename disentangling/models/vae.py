from typing import List, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .ae import AE


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    Reparameterization trick to sample from N(mu, var) from N(0,1).
    """
    # small note for why it compute std like this:
    # https://stats.stackexchange.com/questions/486158/reparameterization-trick-in-vaes-how-should-we-do-this
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


class VAE(AE):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encoder_output_shape: Tuple,
        decoder_input_shape: Tuple,
        latent_dim: int,
    ) -> None:
        super().__init__(encoder, decoder)
        # compatible layer between encoder and latent space
        self.post_encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(
                torch.prod(torch.tensor(encoder_output_shape)), latent_dim * 2
            ),
        )
        # compatible layer between decoder and latent space
        self.pre_decoder = nn.Sequential(
            nn.Linear(
                latent_dim, torch.prod(torch.tensor(decoder_input_shape))
            ),
            nn.Unflatten(1, decoder_input_shape),
        )
        self.latent_dim = latent_dim

    @property
    def latent_space(self):
        return (self.latent_dim,)

    def encode(self, input: Tensor) -> Tensor:
        encoded = self.encoder(input)
        encoded = self.post_encoder(encoded)
        latent_dim = encoded.shape[1] // 2
        return encoded[:, :latent_dim]

    def decode(self, input: Tensor) -> Tensor:
        decoded = self.pre_decoder(input)
        decoded = self.decoder(decoded)
        return decoded

    def forward(self, input: Tensor) -> List[Tensor]:
        encoded = self.encoder(input)
        encoded = self.post_encoder(encoded)
        latent_dim = encoded.shape[1] // 2
        mu, logvar = encoded[:, :latent_dim], encoded[:, latent_dim:]
        z = reparameterize(mu, logvar)
        decoded = self.pre_decoder(z)
        decoded = self.decoder(decoded)
        return decoded, mu, logvar, z

    def loss_function(self, input: Tensor, output: Tensor, *args) -> dict:
        decoded, mu, logvar, *_ = output
        batch_size = decoded.shape[0]
        reconstruction_loss = (
            F.mse_loss(input, decoded, reduction="sum") / batch_size
        )
        kld_loss = (
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        ) / batch_size

        loss = reconstruction_loss + kld_loss
        return dict(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            kld_loss=kld_loss,
        )
