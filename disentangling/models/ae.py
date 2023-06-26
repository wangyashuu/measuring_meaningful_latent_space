from torch import Tensor, nn

from ..utils.loss import get_reconstruction_loss


class AE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        """Create an Autoencoder model with the input encoder and decoder."""
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, input: Tensor) -> Tensor:
        """Encode the input to latents."""
        encoded = self.encoder(input)
        return encoded

    def decode(self, input: Tensor) -> Tensor:
        """Decode the input to sample."""
        decoded = self.decoder(input)
        return decoded

    def forward(self, input: Tensor) -> Tensor:
        """Pass the input through the encoder and the decoder."""
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return decoded


def compute_ae_loss(
    input: Tensor, ae: AE, distribution: str = "bernoulli", *args, **kwargs
) -> dict:
    """Compute the input autoencoder loss.

    Args:
        input (torch.Tensor): The input tensor
        ae (torch.nn.Module): Autoendoer model that accept the shape same as the input.
        distribution (string, optional): String in ["bernoulli", "gaussian"] describe the distribution of input sample, which will effect the reconstruction loss calculation: "bernoulli" will use BCE loss, while "gaussian" will use MSE loss. Default: "bernoulli".

    Returns:
        dict: The dict with loss name (string) as key and loss value (Tensor) as value.
    """

    output = ae(input)
    decoded = output
    reconstruction_loss = get_reconstruction_loss(decoded, input, distribution)
    loss = reconstruction_loss
    return dict(loss=loss)
