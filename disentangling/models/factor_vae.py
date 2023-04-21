from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .vae import VAE
from ..utils.loss import (
    get_reconstruction_loss,
    get_kld_loss,
    get_kld_decomposed_losses,
)


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
    distribution="bernoulli",
    *args,
    **kwargs,
):
    output = factor_vae(input)
    decoded, mu, logvar, z = output

    reconstruction_loss = get_reconstruction_loss(decoded, input, distribution)
    kld_loss = get_kld_loss(mu, logvar)

    (
        mutual_info_loss,
        tc_loss,
        dimension_wise_kl_loss,
    ) = get_kld_decomposed_losses(
        z, mu, logvar, dataset_size=kwargs.pop("dataset_size")
    )

    ###
    # D(z) is the prob of the input from q(z) rather than from ̄q(z)
    # logits is the logits that used to computed the probabilty.
    #   - it is direct output, without normalized (means \sum_{i} sigmoid(logit_i) is not 1)
    #   - logit_qz the logit that the input from q(z)
    # D(z) = softmax(logits)[qz] = exp(logit_qz) / (exp(logit_qz) + exp(logit_q̄z))
    # 1 - D(z) = softmax(logits)[q̄z] = exp(logit_q̄z) / (exp(logit_qz) + exp(logit_q̄z))
    # log(D(z) / 1 - D(z)) = logit_qz - logit_q̄z
    ###
    z_logits = factor_vae_discriminator(z)
    discriminator_tc_loss = torch.mean(z_logits[:, 0] - z_logits[:, 1])
    loss = (
        reconstruction_loss + kld_loss + tc_loss_factor * discriminator_tc_loss
    )
    return dict(
        loss=loss,
        reconstruction_loss=reconstruction_loss,
        kld_loss=kld_loss,
        discriminator_tc_loss=discriminator_tc_loss,
        mutual_info_loss=mutual_info_loss,
        tc_loss=tc_loss,
        dimension_wise_kl_loss=dimension_wise_kl_loss,
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
    decoded, mu, logvar, z = vae_output
    z = z.detach()  # TODO: remove this
    z_permuted = permute_latent(z)
    z_logits = factor_vae_discriminator(z)
    z_permuted_logits = factor_vae_discriminator(z_permuted)
    # encourage z_logit to zero (z_prob[0] to be one)
    discriminator_loss = -0.5 * (
        torch.mean(torch.log(F.softmax(z_logits)[:, 0]))
        + torch.mean(torch.log(F.softmax(z_permuted_logits)[:, 1]))
    )
    return dict(discriminator_loss=discriminator_loss)
