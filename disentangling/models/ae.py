from typing import List, Tuple, Union

from torch import Tensor
from torch.nn import functional as F

from .base_ae import (
    BaseAE,
    get_encoder_fc,
    get_decoder_fc,
)


class AE(BaseAE):
    def __init__(
        self,
        input_shape: Tuple[int],
        hidden_channels: List[int],
        latent_dim: int,
    ) -> None:
        super().__init__(input_shape, hidden_channels)
        self.encoder_fc = get_encoder_fc(
            input_shape=self.encoded_shape, output_dim=latent_dim
        )
        self.decoder_fc = get_decoder_fc(
            input_dim=latent_dim, output_shape=self.encoded_shape
        )

    def encode(self, input: Tensor) -> Tensor:
        encoded = self.encoder_net(input)
        encoded = self.encoder_fc(encoded)
        return encoded

    def decode(self, input: Tensor) -> Tensor:
        decoded = self.decoder_fc(input)
        decoded = self.decoder_net(decoded)
        return decoded

    def forward(self, input: Tensor) -> Tensor:
        encoded = self.encoder_net(input)
        encoded = self.encoder_fc(encoded)
        decoded = self.decoder_fc(encoded)
        decoded = self.decoder_net(decoded)
        return decoded

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
