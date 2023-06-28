from typing import List, Tuple

import torch
from torch import nn, Tensor

from .ae import AE
from ..utils.loss import get_reconstruction_loss, get_kld_loss


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """Reparameterization trick to sample from N(mu, var) to N(0, 1).
    
    Args:
        mu (torch.nn.Tensor): The excepted mean of gaussian distribution.
        var (torch.nn.Tensor): The excepted variance of gaussian distribution.
    
    Returns:
        Tensor: the sampled value
    """
    # small note for why it compute std like this:
    # https://stats.stackexchange.com/questions/486158/reparameterization-trick-in-vaes-how-should-we-do-this
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


class VAE(AE):
    """Variational autoencoder.

    Attributes:
        latent_space (Tuple[int]): the shape of the latent space.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encoder_output_shape: Tuple[int],
        decoder_input_shape: Tuple[int],
        latent_dim: int,
    ) -> None:
        """Create a Variational autoencoder model with the input encoder and decoder.

        Args:
            enoder (torch.nn.Module): The input encoder model.
            decoder (torch.nn.Module): The input decoder
            encoder_output_shape (Tuple[int]): The output shape of the input encoder model, it will be use to create the compatible layer between corresponding encoder and the latent space.
            decoder_input_shape (Tuple[int]): The input shape of the input decoder model, it will be use to create the compatible layer between the latent space and the corresponding encoder.
            latent_dim (int): dimensionality of the latent space.
        """

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
            nn.ReLU(),
            nn.Unflatten(1, decoder_input_shape),
        )
        self.latent_dim = latent_dim

    @property
    def latent_space(self):
        return (self.latent_dim,)

    def encode(self, input: Tensor) -> Tensor:
        """Encode the input to latents."""
        encoded = self.encoder(input)
        encoded = self.post_encoder(encoded)
        latent_dim = encoded.shape[1] // 2
        return encoded[:, :latent_dim]

    def decode(self, input: Tensor) -> Tensor:
        """Decode the input to sample."""
        decoded = self.pre_decoder(input)
        decoded = self.decoder(decoded)
        return decoded

    def forward(self, input: Tensor) -> List[Tensor]:
        """Pass the input through the encoder, the compatible layer of encoder, the compatible layer of decoder and the decoder."""
        encoded = self.encoder(input)
        encoded = self.post_encoder(encoded)
        latent_dim = encoded.shape[1] // 2
        mu, logvar = encoded[:, :latent_dim], encoded[:, latent_dim:]
        z = reparameterize(mu, logvar)
        decoded = self.pre_decoder(z)
        decoded = self.decoder(decoded)
        return decoded, mu, logvar, z


def compute_vae_loss(
    input: Tensor,
    vae: VAE,
    distribution: str = "bernoulli",
    *args,
    **kwargs
) -> dict:
    """Compute the input Variational autoencoder loss.

    Learning object of VAE from `Auto-Encoding Variational Bayes <https://arxiv.org/abs/1312.6114>`

    Args:
        input (torch.nn.Tensor): The input tensor.
        vae (torch.nn.Module): Variational autoendoer model that accept the shape same as the input.
        distribution (string): String in ["bernoulli", "gaussian"] describe the distribution of input sample, which will effect the reconstruction loss calculation: "bernoulli" will use BCE loss, while "gaussian" will use MSE loss.

    Returns:
        dict: The dict with loss name (string) as key and loss value (Tensor) as value, where
            - "loss" represents the total loss,
            - "reconstruction_loss" represents the $\mathbb{E}_{\hat p(x)}[\mathbb{E}_{z \sim q_{\phi}(z|x)} [- \log p_{\theta}(x|z)]]$,
            - "kld_loss" represents $D_{KL} ({q_{\phi}(z | x^{(i)})} | {p_{\theta}(z)})$.
    """
    output = vae(input)
    decoded, mu, logvar, *_ = output
    reconstruction_loss = get_reconstruction_loss(decoded, input, distribution)
    kld_loss = get_kld_loss(mu, logvar)
    loss = reconstruction_loss + kld_loss
    return dict(
        loss=loss,
        reconstruction_loss=reconstruction_loss,
        kld_loss=kld_loss,
    )
