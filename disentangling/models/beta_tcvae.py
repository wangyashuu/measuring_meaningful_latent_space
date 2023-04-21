from typing import Tuple

import torch
from torch import Tensor, nn
from torch.distributions.normal import Normal

from .vae import VAE
from ..utils.loss import (
    get_reconstruction_loss,
    get_kld_loss,
    get_kld_decomposed_losses,
)


def log_density_gaussian(x: Tensor, mu: Tensor, logvar: Tensor):
    import math

    """
    Computes the log pdf of the Gaussian with parameters mu and logvar at x
    :param x: (Tensor) Point at whichGaussian PDF is to be evaluated
    :param mu: (Tensor) Mean of the Gaussian distribution
    :param logvar: (Tensor) Log variance of the Gaussian distribution
    :return:
    """
    norm = -0.5 * (math.log(2 * math.pi) + logvar)
    log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
    return log_density


# def log_prob(value, mu, std):
#     # compute the variance
#     var = std**2
#     log_scale = torch.log(std)
#     return (
#         -((value - mu) ** 2) / (2 * var)
#         - log_scale
#         - math.log(math.sqrt(2 * math.pi))
#     )


class BetaTCVAE(VAE):
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


def compute_beta_tcvae_loss(
    input,
    beta_tcvae,
    minibatch_stratified_sampling,
    mutual_info_loss_factor,
    tc_loss_factor,
    dimension_wise_kl_loss_factor,
    dataset_size=None,
    distribution="bernoulli",
    step=0,
    beta=None,
    *args,
    **kwargs,
) -> dict:
    output = beta_tcvae(input)
    decoded, mu, logvar, z = output

    reconstruction_loss = get_reconstruction_loss(decoded, input, distribution)
    kld_loss = get_kld_loss(mu, logvar)  # for log

    (
        mutual_info_loss,
        tc_loss,
        dimension_wise_kl_loss,
    ) = get_kld_decomposed_losses(
        z,
        mu,
        logvar,
        dataset_size=dataset_size,
        minibatch_stratified_sampling=minibatch_stratified_sampling,
    )

    if beta is not None:
        loss = reconstruction_loss + beta * kld_loss
        return dict(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            kld_loss=kld_loss,
            mutual_info_loss=mutual_info_loss,
            tc_loss=tc_loss,
            dimension_wise_kl_loss=dimension_wise_kl_loss,
        )

    loss = (
        reconstruction_loss
        + mutual_info_loss_factor * mutual_info_loss
        + tc_loss_factor * tc_loss
        + dimension_wise_kl_loss_factor * dimension_wise_kl_loss
    )

    return dict(
        loss=loss,
        reconstruction_loss=reconstruction_loss,
        mutual_info_loss=mutual_info_loss,
        tc_loss=tc_loss,
        dimension_wise_kl_loss=dimension_wise_kl_loss,
        kld_loss=kld_loss,
    )
