import torch
from torch import Tensor

from .vae import VAE
from ..utils.loss import get_reconstruction_loss, get_kld_loss


class DIPVAE(VAE):
    pass


def compute_dip_vae_loss(
    input: Tensor,
    dip_vae: DIPVAE,
    dip_type: str,
    lambda_od: float,
    lambda_d: float,
    distribution: str = "bernoulli",
    pixel_level: bool = False,
    beta: float = 1,
    *args,
    **kwargs,
) -> dict:
    """Compute the DIPVAE loss.

    Learning object of DIPVAE from `Variational Inference of Disentangled Latent Concepts from Unlabeled Observations <https://arxiv.org/abs/1711.00848>`
    Learning object of BetaVAE from `Learning Basic Visual Concepts with a Constrained Variational Framework <https://openreview.net/forum?id=Sy2fzU9gl>`

    Args:
        input (torch.Tensor): The input tensor
        dip_vae (DIPVAE): DIPVAE model that accept the shape same as the input.
        dip_type: string in ["i", "ii"] describe the type of DIPVAE.
        lambda_od (float): Parameter for off diagnal distance in the learning object of DIPVAE.
        lambda_d (float): Parameter for diagnal distance in the learning object of DIPVAE.
        distribution (str): String in ["bernoulli", "gaussian"] describe the distribution of input sample, which will effect the reconstruction loss calculation: "bernoulli" will use BCE loss, while "gaussian" will use MSE loss.
        pixel_level (bool, optional): If divide the VAE loss into pixel level. Default: False.
        beta (float, optional): Parameter for $D_{KL} ({q_{\phi}(z | x^{(i)})} | {p_{\theta}(z)})$ in the learning object of BetaVAE.

    Returns:
        dict: The dict with loss name (string) as key and loss value (Tensor) as value, where
            - "loss" represents the total loss,
            - "reconstruction_loss" represents the $\mathbb{E}_{\hat p(x)}[\mathbb{E}_{z \sim q_{\phi}(z|x)} [- \log p_{\theta}(x|z)]]$,
            - "kld_loss" represents $D_{KL} ({q_{\phi}(z | x^{(i)})} | {p_{\theta}(z)})$.
            - "dip_off_diag_loss" represents the loss of off diagnal distance,
            - "dip_diag_loss" represents the loss of diagnal distance,
            - "dip_loss" represents the sum of dip_off_diag_loss and dip_diag_loss
    """

    output = dip_vae(input)
    decoded, mu, logvar, z = output

    reconstruction_loss = get_reconstruction_loss(decoded, input, distribution)
    kld_loss = get_kld_loss(mu, logvar)

    # cov(mu) = E((mu - E(mu))**2) = E(mu*mu^T) - E(mu)*E(mu)^T
    centered_mu = mu - mu.mean(0)
    cov_mu = (centered_mu.T @ centered_mu) / input.shape[0]

    target_cov = None
    if dip_type == "i":
        target_cov = cov_mu
    elif dip_type == "ii":
        # expectation_sigma = torch.sqrt(logvar.exp()).mean(0)
        # sigma = torch.diag_embed(expectation_sigma, offset=0, dim1=-2, dim2=-1)
        # cov_z = sigma + cov_mu
        cov = torch.diag_embed(logvar.exp(), offset=0, dim1=-2, dim2=-1)
        expectation_cov = torch.mean(cov, dim=0)
        cov_z = expectation_cov + cov_mu
        target_cov = cov_z
    else:
        raise NotImplementedError(
            f"compute_dip_vae_loss dip_type = {dip_type}"
        )
    diag = torch.diagonal(target_cov, offset=0, dim1=-2, dim2=-1)
    off_diag = target_cov - torch.diag_embed(diag, offset=0, dim1=-2, dim2=-1)
    dip_off_diag_loss = torch.sum(off_diag**2)
    dip_diag_loss = torch.sum((diag - 1) ** 2)
    dip_loss = lambda_od * dip_off_diag_loss + lambda_d * dip_diag_loss
    if pixel_level:
        n_dims = torch.prod(torch.tensor(input.shape[1:]))
        loss = (reconstruction_loss + beta * kld_loss) / n_dims + dip_loss
    else:
        loss = reconstruction_loss + beta * kld_loss + dip_loss

    return dict(
        loss=loss,
        reconstruction_loss=reconstruction_loss,
        kld_loss=kld_loss,
        dip_off_diag_loss=dip_off_diag_loss,
        dip_diag_loss=dip_diag_loss,
        dip_loss=dip_loss,
    )
