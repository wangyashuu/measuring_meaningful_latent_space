from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .vae import VAE


def permute_latent(z):
    permuted = []
    n_latent_dim = z.shape[1]
    for i in range(n_latent_dim):
        permuted_ids = torch.randperm(z.shape[0]).to(z.device)
        permuted.append(z[permuted_ids, i])
    return torch.stack(permuted, dim=1)


class FactorVAE(VAE):
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


def compute_factor_vae_loss(
    input,
    factor_vae,
    factor_vae_discriminator,
    tc_loss_factor,
    *args,
    **kwargs,
):
    output = factor_vae(input)
    decoded, mu, logvar, z = output
    batch_size = decoded.shape[0]
    reconstruction_loss = (
        F.mse_loss(input, decoded, reduction="sum") / batch_size
    )
    kld_loss = (
        -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    ) / batch_size
    z_logits = factor_vae_discriminator(z)
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
    # loss = reconstruction_loss + kld_loss + tc_loss_factor * tc_loss
    loss = reconstruction_loss + kld_loss + tc_loss_factor * tc_loss
    return dict(
        loss=loss,
        reconstruction_loss=reconstruction_loss,
        kld_loss=kld_loss,
        tc_loss=tc_loss,
    )


class FactorVAEDiscriminator(nn.Module):
    def __init__(self, latent_dim) -> None:
        super().__init__()
        ###
        # learn to the density ratio needed for estimating TC.
        # output[0] an the probability that its input is a sample from q(z) rather than ̄q(z).
        # output[1] an the probability that its input is a sample from ̄q(z) rather than q(z).
        ###
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.backbone(input)


def compute_factor_vae_discriminator_loss(
    input, factor_vae, factor_vae_discriminator, *args, **kwargs
):
    vae_output = factor_vae(input)
    batch_size = input.shape[0]
    ones = torch.ones(batch_size, dtype=torch.long, device=input.device)
    zeros = torch.zeros(batch_size, dtype=torch.long, device=input.device)
    decoded, mu, logvar, z = vae_output
    z = z.detach()  # TODO: remove this
    z_permuted = permute_latent(z)
    # z_logits_all = factor_vae_discriminator(torch.vstack([z, z_permuted]))
    # z_logits = z_logits_all[:batch_size]
    # z_logits_permuted = z_logits_all[batch_size:]

    z_logits, z_logits_permuted = (
        factor_vae_discriminator.backbone(z),
        factor_vae_discriminator.backbone(z_permuted),
    )
    # encourage z_logit to zero (z_prob[0] to be one)
    discriminator_loss = 0.5 * (
        F.cross_entropy(z_logits, zeros)  # => encourage [1, 0]
        + F.cross_entropy(z_logits_permuted, ones)  # encourage [0, 1]
    )
    return dict(discriminator_loss=discriminator_loss)
