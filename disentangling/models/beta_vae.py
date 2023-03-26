from typing import List, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .vae import VAE


class BetaVAE(VAE):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encoder_output_shape: Tuple,
        decoder_input_shape: Tuple,
        latent_dim: int,
        beta: int,
        c_max = None,
        n_c_iter = None,
    ) -> None:
        super().__init__(
            encoder,
            decoder,
            encoder_output_shape,
            decoder_input_shape,
            latent_dim,
        )
        self.beta = beta
        self.c_max = c_max
        self.n_c_iter = n_c_iter

    def loss_function(
        self, input: Tensor, output: Union[Tensor, List[Tensor]], *args, **kwargs
    ) -> dict:
        decoded, mu, logvar, *_ = output
        batch_size = decoded.shape[0]
        reconstruction_loss = (
            F.mse_loss(input, decoded, reduction="sum") / batch_size
        )
        kld_loss = (
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        ) / batch_size

        c = torch.tensor(0).to(reconstruction_loss)
        if self.c_max is not None:
            n_iter = kwargs.pop('n_iter')
            c = (n_iter / self.n_c_iter) * self.c_max
            # c = torch.clamp(c, 0, self.c_max)
            loss = reconstruction_loss + self.beta * (kld_loss - c).abs()
        else:
            loss = reconstruction_loss + self.beta * kld_loss

        return dict(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            kld_loss=kld_loss,
            c=c
        )
