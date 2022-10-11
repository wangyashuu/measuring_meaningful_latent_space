from typing import List, Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .vae import VAE


class DIPVAE(VAE):
    def __init__(
        self,
        input_shape: Tuple[int],
        hidden_channels: List[int],
        latent_dim: int,
        dip_type: str = "i",
        lambda_od: float = 1.0,
        lambda_d_factor: float = 1.0,
    ) -> None:
        super().__init__(input_shape, hidden_channels, latent_dim)
        self.dip_type = dip_type
        self.lambda_od = lambda_od
        self.lambda_d_factor = lambda_d_factor

    def loss_function(
        self, input: Tensor, output: Union[Tensor, List[Tensor]], *args
    ) -> dict:
        decoded, mu, logvar, z = output
        batch_size = decoded.shape[0]
        reconstruction_loss = (
            F.mse_loss(input, decoded, reduction="sum") / batch_size
        )

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1),
            dim=0,
        )
        expectation_mu_mu_t = (mu[:, :, None] @ mu[:, None, :]).mean(0)
        expectation_mu = mu.mean(0)
        # TODO: how to compute covariance
        cov_mu = (
            expectation_mu_mu_t
            - expectation_mu[None, :] @ expectation_mu[:, None]
        )
        lambda_d = self.lambda_d_factor * self.lambda_od
        target_cov = None
        if self.dip_type == "i":
            target_cov = cov_mu
        elif self.dip_type == "ii":
            cov = torch.diag_embed(logvar.exp(), offset=0, dim1=-2, dim2=-1)
            expectation_cov = torch.mean(cov, dim=0)
            cov_z = expectation_cov + cov_mu
            target_cov = cov_z
        else:
            raise NotImplementedError("DIP variant not supported.")
        diag = torch.diagonal(target_cov, offset=0, dim1=-2, dim2=-1)
        off_diag = diag - torch.diag_embed(diag, offset=0, dim1=-2, dim2=-1)
        dip_loss = self.lambda_od * torch.sum(off_diag**2) \
            + lambda_d * torch.sum((diag - 1) ** 2)

        loss = reconstruction_loss + kld_loss + dip_loss
        return dict(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            kld_loss=kld_loss,
            dip_loss=dip_loss,
        )
