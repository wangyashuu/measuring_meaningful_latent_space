from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .vae import VAE
from torch.distributions.normal import Normal


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


def log_importance_weight_matrix(batch_size, dataset_size):
    """
    Code from (https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py)
    """

    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[:: M + 1] = 1 / N
    W.view(-1)[1 :: M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()


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
    dataset_size,
    *args,
    **kwargs,
) -> dict:
    output = beta_tcvae(input)
    decoded, mu, logvar, z = output

    batch_size = decoded.shape[0]
    std = torch.exp(0.5 * logvar)

    reconstruction_loss = (
        F.mse_loss(input, decoded, reduction="sum") / batch_size
    )

    log_p_z = Normal(0, 1).log_prob(z).sum(-1)
    log_q_z_given_x = Normal(mu, std).log_prob(z).sum(-1)

    # \log{q(z)} ~= -\log{NM} + 1/M \sum^M_i \log{ \sum^M_j q( z_x^{(i)} | x^{(j)} ) }
    # \log{ q( z_x^{(i)} | x^{(j)} ) } where z_x^{(i)} is a sample from q(z|x^{(i)})
    # [batch_size, batch_size, latent_dim], [i, j] => probs of z_xi given xj
    log_q_z_xi_given_xj = Normal(
        mu.reshape(1, batch_size, -1),  # => [1, batch_size, latent_dim]
        std.reshape(1, batch_size, -1),
    ).log_prob(
        z.reshape(batch_size, 1, -1)  # => [batch_size, 1, latent_dim]
    )

    if minibatch_stratified_sampling:
        logiw_matrix = log_importance_weight_matrix(
            batch_size, dataset_size
        ).to(input.device)

        log_q_z = torch.logsumexp(
            logiw_matrix + log_q_z_xi_given_xj.sum(-1), dim=-1
        )
        log_prod_q_z = torch.logsumexp(
            logiw_matrix.reshape(batch_size, batch_size, -1)
            + log_q_z_xi_given_xj,
            dim=1,
        ).sum(-1)

    else:
        log_mn = torch.log(
            torch.tensor([batch_size * dataset_size], device=input.device)
        )
        log_q_z = -log_mn + torch.logsumexp(
            log_q_z_xi_given_xj.sum(-1), dim=-1
        )
        log_prod_q_z = (
            -log_mn + torch.logsumexp(log_q_z_xi_given_xj, dim=1)
        ).sum(-1)

    mutual_info_loss = (log_q_z_given_x - log_q_z).mean()
    tc_loss = (log_q_z - log_prod_q_z).mean()
    dimension_wise_kl_loss = (log_prod_q_z - log_p_z).mean()
    loss = (
        reconstruction_loss
        + mutual_info_loss_factor * mutual_info_loss
        + tc_loss_factor * mutual_info_loss
        + dimension_wise_kl_loss_factor * dimension_wise_kl_loss
    )
    return dict(
        loss=loss,
        reconstruction_loss=reconstruction_loss,
        mutual_info_loss=mutual_info_loss,
        tc_loss=tc_loss,
        dimension_wise_kl_loss=dimension_wise_kl_loss,
    )
