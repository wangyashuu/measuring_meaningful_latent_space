from typing import List, Tuple, Union

from torch import Tensor
from torch.nn import functional as F

from .base_ae import BaseAE


class AE(BaseAE):
    def __init__(
        self,
        input_shape: Tuple[int],
        hidden_channels: List[int],
        latent_dim: int,
    ) -> None:
        super().__init__(
            encoder_input_shape=input_shape,
            encoder_output_dim=latent_dim,
            decoder_input_dim=latent_dim,
            hidden_channels=hidden_channels,
        )

    def forward(self, input: Tensor) -> Tensor:
        encoder_out = self.encoder(input)
        decoder_out = self.decoder(encoder_out)
        return decoder_out

    def loss_function(
        self, input: Tensor, output: Union[Tensor, List[Tensor]], **kwargs
    ) -> dict:
        decoded = output
        batch_size = decoded.shape[0]
        reconstruction_loss = (
            F.mse_loss(input, decoded, reduction="sum") / batch_size
        )
        loss = reconstruction_loss
        return loss
