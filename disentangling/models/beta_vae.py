import torch
from torch import Tensor

from .vae import VAE
from ..utils.loss import get_reconstruction_loss, get_kld_loss


class BetaVAE(VAE):
    pass


def compute_beta_vae_loss(
    input: Tensor,
    beta_vae: BetaVAE,
    distribution: str = "bernoulli",
    beta: float = 1,
    c_max: float = 0,
    n_c_steps: int = 0,
    step: int = 0,
    *args,
    **kwargs
) -> dict:
    """Compute the BetaVAE loss.

    Learning object of BetaVAE from `Learning Basic Visual Concepts with a Constrained Variational Framework <https://openreview.net/forum?id=Sy2fzU9gl>`
    Learning object of Annealing version of BetaVAE from `Understanding disentangling in beta-VAE<https://arxiv.org/abs/1804.03599>`

    Args:
        input (torch.Tensor): The input tensor
        beta_vae (BetaVAE): BetaVAE model that accept the shape same as the input.
        distribution (str): String in ["bernoulli", "gaussian"] describe the distribution of input sample, which will effect the reconstruction loss calculation: "bernoulli" will use BCE loss, while "gaussian" will use MSE loss.
        beta (float): Parameter for $D_{KL} ({q_{\phi}(z | x^{(i)})} | {p_{\theta}(z)})$ in the learning object of BetaVAE.
        c_max (float, optional): Parameter in the learning object of Annealing version of BetaVAE. Default: 0.
        n_c_steps (int, optional): Parameter in the learning object of Annealing version of BetaVAE. Default: 0.
        c_max (int, optional): The current global step, parameter of the learning object of Annealing version of BetaVAE. Default: 0.

    Returns:
        dict: The dict with loss name (string) as key and loss value (Tensor) as value, where
            - "loss" represents the total loss,
            - "reconstruction_loss" represents the $\mathbb{E}_{\hat p(x)}[\mathbb{E}_{z \sim q_{\phi}(z|x)} [- \log p_{\theta}(x|z)]]$,
            - "kld_loss" represents $D_{KL} ({q_{\phi}(z | x^{(i)})} | {p_{\theta}(z)})$.
    """

    output = beta_vae(input)
    decoded, mu, logvar, *_ = output
    reconstruction_loss = get_reconstruction_loss(decoded, input, distribution)
    kld_loss = get_kld_loss(mu, logvar)

    if c_max > 0:
        c_max = torch.tensor(c_max, device=input.device)
        c = torch.clamp(c_max * (step / n_c_steps), 0, c_max)
        loss = reconstruction_loss + beta * (kld_loss - c).abs()
        return dict(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            kld_loss=kld_loss,
            c=c,
        )

    loss = reconstruction_loss + beta * kld_loss
    return dict(
        loss=loss,
        reconstruction_loss=reconstruction_loss,
        kld_loss=kld_loss,
    )
