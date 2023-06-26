from typing import Tuple

import torch
from torch import Tensor, nn

from .vae import VAE
from ..utils.loss import get_reconstruction_loss, get_kld_loss


def rbf_kernel(z1, z2, sigma):
    """Compute the rbf kernel.

    `Radial basis function kernel<https://en.wikipedia.org/wiki/Radial_basis_function_kernel>`
    """
    n_latent_dim = z1.shape[1]
    k = torch.exp(
        -torch.mean((z1[:, None, :] - z2[None, :, :]) ** 2, dim=-1)
        / (1.0 * n_latent_dim * sigma**2)
    )
    return k


def init_weights(m):
    if (
        isinstance(m, nn.Linear)
        or isinstance(m, nn.ConvTranspose2d)
        or isinstance(m, nn.Conv2d)
    ):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class InfoVAE(VAE):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.apply(init_weights)


def compute_info_vae_loss(
    input: Tensor,
    info_vae: InfoVAE,
    alpha: float,
    lambd: float,
    kernel_type: str = "rbf",
    kernel_sigma: float = 1,
    distribution="bernoulli",
    *args,
    **kwargs
) -> dict:
    """Compute the InfoVAE loss.

    Learning object of InfoVAE from `InfoVAE: Information Maximizing Variational Autoencoders <https://arxiv.org/abs/1706.02262>`
    Reference official implementation `Tensorflow Implementation of MMD Variational Autoencoder <https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder/blob/master/mmd_vae.ipynb>`

    Args:
        input (torch.Tensor): The input tensor
        info_vae (InfoVAE): InfoVAE model that accept the shape same as the input.
        alpha (float): Parameter for controlling the kld loss in the learning object of InfoVAE, where the kld loss factor is 1 - alpha).
        lambd (float): Parameter for controlling the mmd loss in the learning object of InfoVAE, where the mmd loss factor is alpha + lambd - 1.
        distribution (str, optional): String in ["bernoulli", "gaussian"] describe the distribution of input sample, which will effect the reconstruction loss calculation: "bernoulli" will use BCE loss, while "gaussian" will use MSE loss. Default: "bernoulli".
        kernel_type (str, optional): String in ["rbf", "imq"] decribe the kernel type, currently only support rbf kernel. Default: "rbf".
        kernel_sigma (float, optional): Parameter for computing the kernel. Default: 1.

    Returns:
        dict: the dict with loss name (string) as key and loss value (Tensor) as value, where
            - "loss" represents the total loss,
            - "reconstruction_loss" represents $\mathbb{E}_{\hat p(x)}[\mathbb{E}_{z \sim q_{\phi}(z|x)} [- \log p_{\theta}(x|z)]]$,
            - "mutual_info_loss" represents $I_{q_\phi}(x;z)$,
            - "tc_loss" represents $\textrm{TC}(q_\phi(z))$,
            - "dimension_wise_kl_loss" represents $\sum_j D_{\textrm{KL}}(q_\phi(z_j) || p(z_j))$,
            - "kld_loss" represents $D_{KL} ({q_{\phi}(z | x^{(i)})} | {p_{\theta}(z)})$.
    """

    # TODO: refactor
    # kld_loss_factor = 1 - alpha
    # mmd_loss_factor = gamma - (1-alpha)

    output = info_vae(input)
    decoded, mu, logvar, z = output

    reconstruction_loss = get_reconstruction_loss(decoded, input, distribution)
    kld_loss = get_kld_loss(mu, logvar)

    z_prior = torch.randn_like(z, device=z.device)
    z_posterior = z
    compute_kernel = rbf_kernel if kernel_type == "rbf" else imq_kernel
    z_prior_kernel = compute_kernel(z_prior, z_prior, sigma=kernel_sigma)
    z_posterior_kernel = compute_kernel(
        z_posterior, z_posterior, sigma=kernel_sigma
    )
    cross_kernel = compute_kernel(z_prior, z_posterior, sigma=kernel_sigma)
    mmd_loss = (
        z_posterior_kernel.mean()
        + z_prior_kernel.mean()
        - 2 * cross_kernel.mean()
    )
    n_dims = torch.prod(torch.tensor(input.shape[1:]))
    loss = (
        (reconstruction_loss / n_dims)
        + (1 - alpha) * (kld_loss / n_dims)
        + (alpha + lambd - 1) * mmd_loss
    )

    return dict(
        loss=loss,
        reconstruction_loss=reconstruction_loss,
        kld_loss=kld_loss,
        mmd_loss=mmd_loss,
    )
