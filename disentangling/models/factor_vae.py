import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .vae import VAE
from ..utils.loss import get_reconstruction_loss, get_kld_loss


def permute_latent(z):
    """Permute the latent values in each latent dimension."""
    permuted = []
    n_latent_dim = z.shape[1]
    for i in range(n_latent_dim):
        permuted_ids = torch.randperm(z.shape[0]).to(z.device)
        permuted.append(z[permuted_ids, i])
    return torch.stack(permuted, dim=1)


class FactorVAE(VAE):
    pass


class FactorVAEDiscriminator(nn.Module):
    """Discriminator of FactorVAE.

    learn to the density ratio needed for estimating TC. where
        - output[0] an the probability that its input is a sample from $q(z)$ rather than $\bar q(z)$.
        - output[1] an the probability that its input is a sample from $\bar q(z)$ rather than $q(z)$.
    """

    def __init__(self, latent_dim: int) -> None:
        """Creates the discriminator backbone

        Args:
            latent_dim (int): the latent dimentionality of the corresponding FactorVAE
        """

        super().__init__()
        ###
        # issuse when use batch norm in ddp and forward two times:
        # ref: (https://github.com/pytorch/pytorch/issues/66504, https://github.com/pytorch/pytorch/issues/73332, https://github.com/pytorch/pytorch/issues/26288)
        # please set track_running_stats = false
        ###
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 2),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.backbone(input)


def compute_factor_vae_loss(
    input: Tensor,
    factor_vae: FactorVAE,
    factor_vae_discriminator: FactorVAEDiscriminator,
    d_tc_loss_factor: float,
    distribution: str = "bernoulli",
    *args,
    **kwargs,
) -> dict:
    """Compute the FactorVAE loss.

    Learning object of FactorVAE from `Disentangling by Factorizing <https://arxiv.org/abs/1802.05983>.`

    Args:
        input (torch.Tensor): The input tensor
        factor_vae (FactorVAE): Factor model that accept the shape same as the input.
        factor_vae_discriminator (FactorVAEDiscriminator): discriminator model of the corresponding FactorVAE.
        d_tc_loss_factor (float): Parameter for $\mathrm{TC}(q_\phi(z))$ in the learning object of FactorVAE.
        distribution (str): string in ["bernoulli", "gaussian"] describe the distribution of input sample, which will effect the reconstruction loss calculation: "bernoulli" will use BCE loss, while "gaussian" will use MSE loss.

    Returns:
        dict: the dict with loss name (string) as key and loss value (Tensor) as value, where
            - "loss" represents the total loss,
            - "reconstruction_loss" represents the $\mathbb{E}_{\hat p(x)}[\mathbb{E}_{z \sim q_{\phi}(z|x)} [- \log p_{\theta}(x|z)]]$,
            - "kld_loss" represents $D_{KL} ({q_{\phi}(z | x^{(i)})} | {p_{\theta}(z)})$.
            - "d_tc_loss" represents $\mathrm{TC}(q_\phi(z))$.
    """
    output = factor_vae(input)
    decoded, mu, logvar, z = output

    reconstruction_loss = get_reconstruction_loss(decoded, input, distribution)
    kld_loss = get_kld_loss(mu, logvar)

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
    d_tc_loss = torch.mean(z_logits[:, 0] - z_logits[:, 1])
    loss = reconstruction_loss + kld_loss + d_tc_loss_factor * d_tc_loss
    return dict(
        loss=loss,
        reconstruction_loss=reconstruction_loss,
        kld_loss=kld_loss,
        d_tc_loss=d_tc_loss,
    )


def compute_factor_vae_discriminator_loss(
    input: Tensor,
    factor_vae: FactorVAE,
    factor_vae_discriminator: FactorVAEDiscriminator,
    *args,
    **kwargs,
) -> dict:
    """Compute the FactorVAE discriminator loss.

    Learning object of FactorVAE from `Disentangling by Factorizing <https://arxiv.org/pdf/1802.05983>.`

    Args:
        input (torch.Tensor): The input tensor
        factor_vae (FactorVAE): Factor model that accept the shape same as the input.
        factor_vae_discriminator (FactorVAEDiscriminator): Discriminator model of the corresponding FactorVAE.

    Returns:
        dict: The dict with loss name (string) as key and loss value (Tensor) as value, where
            - "discriminator_loss" represents the total loss,
            - "d_accuracy" represents the accuracy of discriminator (return this value for debug/log).
    """

    vae_output = factor_vae(input)
    decoded, mu, logvar, z = vae_output
    z = z.detach()  # TODO: remove this
    z_permuted = permute_latent(z)
    z_logits = factor_vae_discriminator(z)
    z_permuted_logits = factor_vae_discriminator(z_permuted)
    # encourage z_logit to zero (z_prob[0] to be one)
    z_probs = F.softmax(z_logits, dim=1)[:, 0]
    z_permuted_probs = F.softmax(z_permuted_logits, dim=1)[:, 1]
    discriminator_loss = -0.5 * (
        torch.mean(torch.log(z_probs))
        + torch.mean(torch.log(z_permuted_probs))
    )
    d_accuracy = ((z_probs > 0.5).sum() + (z_permuted_probs > 0.5).sum()) / (
        2 * z_probs.shape[0]
    )
    return dict(discriminator_loss=discriminator_loss, d_accuracy=d_accuracy)
