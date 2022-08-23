from typing import List, Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .vae import VAE
from torch.distributions.normal import Normal


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
        input_shape: Tuple[int],
        hidden_channels: List[int],
        latent_dim: int,
        alpha: float,
        beta: float,
        gamma: float,
        train_set_size: int,
        val_set_size: int,
        minibatch_stratified_sampling: bool = False,
    ) -> None:
        super().__init__(input_shape, hidden_channels, latent_dim)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.train_set_size = train_set_size
        self.val_set_size = val_set_size
        self.minibatch_stratified_sampling = minibatch_stratified_sampling

    def loss_function(
        self, input: Tensor, output: Union[Tensor, List[Tensor]], **kwargs
    ) -> dict:
        decoded, mu, logvar, z = output

        batch_size = decoded.shape[0]
        std = torch.exp(0.5 * logvar)
        dataset_size = (
            self.train_set_size if self.training else self.val_set_size
        )
        device = input.device

        # reconstruction_loss
        # -log_p_x = -Normal(x_mu, x_std).log_prob(input)
        reconstruction_loss = (
            F.mse_loss(input, decoded, reduction="sum") / batch_size
        )

        # why zero sigma here?
        log_p_z = Normal(0, 0).log_prob(z)
        log_q_z_given_x = Normal(mu, std).log_prob(z)

        # \log{q(z)} ~= -\log{NM} + 1/M \sum^M_i \log{ \sum^M_j q( z_x^{(i)} | x^{(j)} ) }
        # \log{ q( z_x^{(i)} | x^{(j)} ) } where z_x^{(i)} is a sample from q(z|x^{(i)})
        # [batch_size, batch_size, latent_dim], [i, j] => probs of z_xi given xj
        log_q_z_xi_given_xj = Normal(
            # => reshape as [1, batch_size, latent_dim]
            mu.reshape(1, batch_size, -1),
            std.reshape(1, batch_size, -1),
        ).log_prob(
            z.reshape(batch_size, 1, -1)
        )  # => reshape as [batch_size, 1, latent_dim]

        if self.minibatch_stratified_sampling:
            logiw_matrix = log_importance_weight_matrix(
                batch_size, dataset_size
            ).to(device)

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
                torch.tensor([batch_size * dataset_size]).to(device)
            )
            log_q_z = -log_mn + torch.logsumexp(
                log_q_z_xi_given_xj.sum(-1), dim=-1
            )
            log_prod_q_z = (
                -log_mn + torch.logsumexp(log_q_z_xi_given_xj, dim=1)
            ).sum(-1)

        mutual_info_loss = log_q_z_given_x - log_q_z
        tc_loss = log_q_z - log_prod_q_z
        dimension_wise_kl = log_prod_q_z - log_p_z

        loss = (
            reconstruction_loss
            + self.alpha * mutual_info_loss
            + self.beta * tc_loss
            + self.gamma * dimension_wise_kl
        )
        return loss
