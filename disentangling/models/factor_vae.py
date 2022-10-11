from typing import List, Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .vae import VAE


def permute_latent(z):
    permuted = torch.zeros_like(z)
    n_latent_dim = z.shape[1]
    for i in range(n_latent_dim):
        permuted_ids = torch.randperm(z.shape[0]).to(z.device)
        permuted[:, i] = z[permuted_ids, i]

    return permuted


class FactorVAE(VAE):
    optimize_VAE = 0
    optimize_discriminator = 1

    def __init__(
        self,
        input_shape: Tuple[int],
        hidden_channels: List[int],
        latent_dim: int,
        tc_loss_factor: float,
        discriminator_dims: List[int] = [1000, 1000, 1000],
    ) -> None:
        super().__init__(input_shape, hidden_channels, latent_dim)
        ###
        # learn to the density ratio needed for estimating TC.
        # output[0] an the probability that its input is a sample from q(z) rather than ̄q(z).
        # output[1] an the probability that its input is a sample from ̄q(z) rather than q(z).
        ###
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 2),
        )
        self.tc_loss_factor = tc_loss_factor

    def vae_loss_function(self, input, output):
        decoded, mu, logvar, z = output
        batch_size = decoded.shape[0]
        reconstruction_loss = (
            F.mse_loss(input, decoded, reduction="sum") / batch_size
        )
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1),
            dim=0,
        )
        z_logits = self.discriminator(z)
        ###
        # D(z) is the prob of the input from q(z) rather than from ̄q(z)
        # logits is the logits that used to computed the probabilty.
        #   - it is direct output, without normalized (means \sum_{i} sigmoid(logit_i) is not 1)
        #   - logit_qz the logit that the input from q(z)
        # D(z) = softmax(logits)[qz] = exp(logit_qz) / (exp(logit_qz) + exp(logit_q̄z))
        # 1 - D(z) = softmax(logits)[q̄z] = exp(logit_q̄z) / (exp(logit_qz) + exp(logit_q̄z))
        # log(D(z) / 1 - D(z)) = logit_qz - logit_q̄z
        ###
        tc_loss = torch.mean(z_logits[:, 0] - z_logits[:, 1])
        loss = reconstruction_loss + kld_loss + self.tc_loss_factor * tc_loss
        return dict(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            kld_loss=kld_loss,
            tc_loss=tc_loss,
        )

    def discriminator_loss_function(self, input, output):
        batch_size = input.shape[0]
        ones = torch.ones(
            batch_size,
            requires_grad=False,
            dtype=torch.long,
            device=input.device,
        )
        zeros = torch.zeros(
            batch_size,
            requires_grad=False,
            dtype=torch.long,
            device=input.device,
        )
        decoded, mu, logvar, z = output
        z = z.detach()  # Detach so that VAE is not trained again
        z_permuted = permute_latent(z)
        z_logits = self.discriminator(z)
        z_permuted_logits = self.discriminator(z_permuted)
        # encourage z_logit to zero (z_prob[0] to be one)
        discriminator_loss = 0.5 * (
            F.cross_entropy(z_logits, zeros) # => encourage [1, 0]
            + F.cross_entropy(z_permuted_logits, ones) # encourage [0, 1]
        )
        return dict(discriminator_loss=discriminator_loss)

    def loss_function(
        self,
        input: Tensor,
        output: Union[Tensor, List[Tensor]],
        optimizer_idx: int = 0,
        *args
    ) -> dict:
        if optimizer_idx == self.optimize_VAE:
            return self.vae_loss_function(input, output)
        elif optimizer_idx == self.optimize_discriminator:
            return self.discriminator_loss_function(input, output)
