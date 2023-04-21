from torch import Tensor, nn

from ..utils.loss import get_reconstruction_loss


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


def compute_ae_loss(
    input, ae, distribution="bernoulli", *args, **kwargs
) -> dict:
    output = ae(input)
    decoded = output
    reconstruction_loss = get_reconstruction_loss(decoded, input, distribution)
    loss = reconstruction_loss
    return dict(loss=loss)
