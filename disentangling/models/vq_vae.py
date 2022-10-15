from typing import List, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .ae import AE
from ..utils.nn import conv2d_output_size


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        n_embeddings: int,
        n_embedding_dims: int,
        commitment_loss_factor: float,
        ema_enabled: bool = True,
        ema_decay: float = 0.99,
        ema_epsilon=1e-5,
    ):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.n_embedding_dims = n_embedding_dims
        self.codebook = nn.Embedding(n_embeddings, n_embedding_dims)
        self.commitment_loss_factor = commitment_loss_factor
        self.ema_enabled = ema_enabled
        if ema_enabled:
            self.codebook.weight.data.normal_()
            self.register_buffer("ema_cluster_size", torch.zeros(n_embeddings))
            self.ema_w = nn.Parameter(
                torch.Tensor(n_embeddings, n_embedding_dims)
            )
            self.ema_w.data.normal_()
            self.ema_decay = ema_decay
            self.ema_epsilon = ema_epsilon
        else:
            self.codebook.weight.data.uniform_(
                -1 / n_embeddings, 1 / n_embeddings
            )

    def update_embeddings(self, encoded_one_hot, flattened):
        decay = self.ema_decay
        epsilon = self.ema_epsilon
        n_embeddings = self.codebook.weight.shape[0]

        self.ema_cluster_size = self.ema_cluster_size * decay + (
            1 - decay
        ) * torch.sum(encoded_one_hot, 0)

        # Laplace smoothing of the cluster size
        n = torch.sum(self.ema_cluster_size.data)
        self.ema_cluster_size = (
            (self.ema_cluster_size + epsilon)
            / (n + n_embeddings * epsilon)
            * n
        )

        dw = encoded_one_hot.t() @ flattened
        self.ema_w = nn.Parameter(self.ema_w * decay + (1 - decay) * dw)
        self.codebook.weight = nn.Parameter(
            self.ema_w / self.ema_cluster_size.unsqueeze(1)
        )

    def forward(self, inputs):
        embeddings = self.codebook.weight
        n_embeddings, n_embedding_dims = embeddings.shape

        latents = inputs.permute(0, 2, 3, 1).contiguous()  # BDHW -> BHWD
        latents_shape = latents.shape
        flattened = latents.view(-1, n_embedding_dims)  # => (BHW, D)

        distances = (
            torch.sum(flattened**2, dim=1, keepdim=True)
            + torch.sum(embeddings**2, dim=1)
        ) - 2 * (flattened @ embeddings.t())
        indices = torch.argmin(distances, dim=1).unsqueeze(1)  # (BHW, 1)

        # look up the embedding in codebook (BHW, N)
        encoded_one_hot = torch.zeros(indices.shape[0], n_embeddings).to(
            latents.device
        )
        encoded_one_hot.scatter_(1, indices, 1)
        quantized_latents = (encoded_one_hot @ embeddings).view(latents_shape)
        quantized = latents + (quantized_latents - latents).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        if self.ema_enabled and self.training:
            self.update_embeddings(encoded_one_hot, flattened)

        return quantized, latents, quantized_latents

    def compute_loss(self, outputs):
        _, latents, quantized_latents = outputs
        # Compute the VQ Losses TODO: what is the different here.
        commitment_loss = self.commitment_loss_factor * F.mse_loss(
            quantized_latents.detach(), latents
        )
        if not self.ema_enabled:
            return commitment_loss
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        return embedding_loss + commitment_loss


class VQVAE(AE):
    """
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=F5hOFwiBmPPh
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encoder_output_shape: Tuple,
        decoder_input_shape: Tuple,
        n_embedding_dims: int,
        n_embeddings: int,
        commitment_loss_factor,
        ema_enabled: bool = True,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
    ) -> None:
        super().__init__(encoder, decoder)
        self.post_encoder = nn.Conv2d(
            in_channels=encoder_output_shape[0],
            out_channels=n_embedding_dims,
            kernel_size=1,
            stride=1,
            padding=1,
        )
        out_size, output_padding = conv2d_output_size(
            encoder_output_shape[1:],
            kernel_size=1,
            stride=1,
            padding=1,
        )
        self.pre_decoder = nn.ConvTranspose2d(
            in_channels=n_embedding_dims,
            out_channels=decoder_input_shape[0],
            kernel_size=1,
            stride=1,
            padding=1,
            output_padding=output_padding,
        )
        self.quantizer = VectorQuantizer(
            n_embeddings,
            n_embedding_dims,
            ema_enabled,
            ema_decay,
            ema_epsilon,
            commitment_loss_factor,
        )
        self.codebook = nn.Embedding(n_embeddings, n_embedding_dims)
        self.commitment_loss_factor = commitment_loss_factor
        self.encoder_output_shape = encoder_output_shape

    @property
    def latent_space(self):
        return (self.n_embedding_dims,) + self.encoder_output_shape[1:]

    def encode(self, inputs: Tensor) -> Tensor:
        encoded = self.encoder(inputs)  # (B, D, H, W)
        encoded = self.post_encoder(encoded)
        quantizer_outputs = self.quantizer(encoded)
        quantized, *_ = quantizer_outputs
        return quantized  # (B, D, H, W)

    def decode(self, inputs: Tensor) -> Tensor:
        decoded = self.pre_decoder(inputs)
        decoded = self.decoder(decoded)
        return decoded

    def forward(self, inputs) -> List[Tensor]:
        encoded = self.encoder(inputs)  # (B, D, H, W)
        encoded = self.post_encoder(encoded)
        quantizer_outputs = self.quantizer(encoded)
        quantized, *_ = quantizer_outputs
        decoded = self.pre_decoder(quantized)
        decoded = self.decoder(decoded)
        return decoded, quantizer_outputs

    def loss_function(self, input, output, *args) -> dict:
        decoded, quantizer_output = output
        batch_size = decoded.shape[0]
        reconstruction_loss = (
            F.mse_loss(input, decoded, reduction="sum") / batch_size
        )
        vq_loss = self.quantizer.compute_loss(quantizer_output)
        loss = reconstruction_loss + vq_loss
        return dict(
            loss=loss, reconstruction_loss=reconstruction_loss, vq_loss=vq_loss
        )

    # TODO: sample
