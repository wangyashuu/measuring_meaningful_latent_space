from torch import Tensor

from .vae import VAE
from ..utils.loss import (
    get_reconstruction_loss,
    get_kld_loss,
    get_kld_decomposed_losses,
)


class BetaTCVAE(VAE):
    pass


def compute_beta_tcvae_loss(
    input: Tensor,
    beta_tcvae: BetaTCVAE,
    mutual_info_loss_factor: float,
    tc_loss_factor: float,
    dimension_wise_kl_loss_factor: float,
    distribution: str = "bernoulli",
    minibatch_stratified_sampling: bool = True,
    dataset_size: int = 0,
    *args,
    **kwargs,
) -> dict:
    """Compute the BetaTCVAE loss.

    Learning object of BetaTCVAE from `Isolating Sources of Disentanglement in Variational Autoencoders <https://arxiv.org/abs/1802.04942>`

    Args:
        input (torch.Tensor): The input tensor
        beta_vae (BetaTCVAE): BetaVAE model that accept the shape same as the input.
        mutual_info_loss_factor (float): Parameter for $I_{q_\phi}(x;z)$ in the learning object of BetaTCVAE.
        tc_loss_factor (float): Parameter for $\textrm{TC}(q_\phi(z))$ in the learning object of BetaTCVAE.
        dimension_wise_kl_loss_factor (string): Parameter for $\sum_j D_{\textrm{KL}}(q_\phi(z_j) || p(z_j))$ in the learning object of BetaTCVAE.
        distribution (str, optional): String in ["bernoulli", "gaussian"] describe the distribution of input sample, which will effect the reconstruction loss calculation: "bernoulli" will use BCE loss, while "gaussian" will use MSE loss. Default: "bernoulli".
        minibatch_stratified_sampling (bool, optional): If using the minibatch stratified sampling for loss calculation. Default: True.
        dataset_size (int, optional): The size of dataset, only available when not using the minibatch stratified sampling. Default: 0.

    Returns:
        dict: The dict with loss name (string) as key and loss value (Tensor) as value, where
            - "loss" represents the total loss,
            - "reconstruction_loss" represents $\mathbb{E}_{\hat p(x)}[\mathbb{E}_{z \sim q_{\phi}(z|x)} [- \log p_{\theta}(x|z)]]$,
            - "mutual_info_loss" represents $I_{q_\phi}(x;z)$,
            - "tc_loss" represents $\textrm{TC}(q_\phi(z))$,
            - "dimension_wise_kl_loss" represents $\sum_j D_{\textrm{KL}}(q_\phi(z_j) || p(z_j))$,
            - "kld_loss" represents $D_{KL} ({q_{\phi}(z | x^{(i)})} | {p_{\theta}(z)})$.
    """

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
