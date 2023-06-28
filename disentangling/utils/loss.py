import torch
from torch import Tensor
from torch.nn import functional as F
from torch.distributions.normal import Normal


def get_reconstruction_loss(
    output: Tensor, input: Tensor, distribution: str = "bernoulli"
):
    """Compute the reconstruction loss for VAE.

    Args:
        output (Tensor): Tensor with any shape.
        input (Tensor): Tensor that has same shape as output.
        distribution (str): String in ["bernoulli", "gaussian"] describe the distribution of input sample, which will effect the reconstruction loss calculation: "bernoulli" will use BCE loss, while "gaussian" will use MSE loss.

    Returns:
        loss (Tensor): Loss of each sample.

    """
    batch_size = input.shape[0]
    if distribution == "bernoulli":
        loss = F.binary_cross_entropy_with_logits(
            output, input, reduction="sum"
        )
    elif distribution == "gaussian":
        loss = F.mse_loss(F.sigmoid(output), input, reduction="sum")
    else:
        raise NotImplementedError(
            f"get_reconstruction_loss distribution = {distribution}"
        )
    return loss / batch_size


def get_kld_loss(mu: Tensor, logvar: Tensor):
    """Compute the kld loss for VAE.
     Args:
        mu (torch.nn.Tensor): The excepted mean of gaussian distribution.
        var (torch.nn.Tensor): The excepted variance of gaussian distribution.

    Returns:
        Tensor: The kld loss of given mean and log variance.
    """
    kld_loss = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(1).mean()
    return kld_loss


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


def get_kld_decomposed_losses(
    z: Tensor,
    mu: Tensor,
    logvar: Tensor,
    dataset_size: int = None,
    minibatch_stratified_sampling: bool = True,
):
    """Calculate the mutual information loss, tc loss and dimension-wise kl loss for BetaTCVAE.

    Code adapted from the offical `BetaTCVAE implementation <https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py>`

    Args:
        z (Tensor): The latent codes of any shape.
        mu (Tensor): The mean of gaussian distribution which is the latent representation $z$ sampled from.
        var (Tensor): The variance of gaussian distribution  which is the latent representation $z$ sampled from.
        dataset_size (int, optional): The size of dataset, only available when not using the minibatch stratified sampling. Default: 0.
        minibatch_stratified_sampling (bool, optional): If using the minibatch stratified sampling for loss calculation. Default: True.

    Returns:
        tuple: Tuple which includes mutual_info_loss, tc_loss, dimension_wise_kl_loss
    """
    batch_size = z.shape[0]
    device = z.device
    std = torch.exp(0.5 * logvar)

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
            torch.tensor([batch_size * dataset_size], device=device)
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
    return mutual_info_loss, tc_loss, dimension_wise_kl_loss
