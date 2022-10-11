from base64 import encode
from typing import List, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .base_ae import BaseAE


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        n_embeddings: int,
        n_embedding_dim: int,
        commitment_loss_factor: float,
        ema_enabled: bool = True,
        ema_decay: float = 0.99,
        ema_epsilon=1e-5,
    ):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.n_embedding_dim = n_embedding_dim
        self.codebook = nn.Embedding(n_embeddings, n_embedding_dim)
        self.commitment_loss_factor = commitment_loss_factor
        self.ema_enabled = ema_enabled
        if ema_enabled:
            self.codebook.weight.data.normal_()
            self.register_buffer("ema_cluster_size", torch.zeros(n_embeddings))
            self.ema_w = nn.Parameter(
                torch.Tensor(n_embeddings, n_embedding_dim)
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
        n_embeddings, n_embedding_dim = embeddings.shape

        latents = inputs.permute(0, 2, 3, 1).contiguous()  # BCHW -> BHWC
        latents_shape = latents.shape
        flattened = latents.view(-1, n_embedding_dim)  # => (BHW, D)

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


class VQVAE(BaseAE):
    """
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=F5hOFwiBmPPh
    """

    def __init__(
        self,
        input_shape: Tuple[int],
        hidden_channels: List[int],
        n_embeddings,
        n_embedding_dim,
        commitment_loss_factor,
        ema_enabled: bool = True,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
    ) -> None:
        super().__init__(input_shape, hidden_channels)
        self.quantizer = VectorQuantizer(
            n_embeddings,
            n_embedding_dim,
            ema_enabled,
            ema_decay,
            ema_epsilon,
            commitment_loss_factor,
        )
        self.codebook = nn.Embedding(n_embeddings, n_embedding_dim)

        self.commitment_loss_factor = commitment_loss_factor

    def forward(self, inputs) -> List[Tensor]:
        encoded = self.encoder_net(inputs)  # (B, D, H, W)
        quantizer_outputs = self.quantizer(encoded)
        quantized, *_ = quantizer_outputs
        decoded = self.decoder_net(quantized)
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
