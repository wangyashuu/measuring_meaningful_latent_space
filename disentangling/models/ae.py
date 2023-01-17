from typing import List, Union

from torch import Tensor, nn
from torch.nn import functional as F


class AE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    # TODO: latent_space

    def encode(self, input: Tensor) -> Tensor:
        encoded = self.encoder(input)
        return encoded

    def decode(self, input: Tensor) -> Tensor:
        decoded = self.decoder(input)
        return decoded

    def forward(self, input: Tensor) -> Tensor:
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return decoded

    def loss_function(
        self, input: Tensor, output: Union[Tensor, List[Tensor]], *args
    ) -> dict:
        decoded = output
        batch_size = decoded.shape[0]
        reconstruction_loss = (
            F.mse_loss(input, decoded, reduction="sum") / batch_size
        )
        loss = reconstruction_loss
        return dict(loss=loss)
