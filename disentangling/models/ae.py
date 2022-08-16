from typing import Optional, List, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .base_ae import BaseAE


def conv2d_output_size(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    from math import floor

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(
        (
            (h_w[0] + (2 * padding) - (dilation * (kernel_size[0] - 1)) - 1)
            / stride
        )
        + 1
    )
    w = floor(
        (
            (h_w[1] + (2 * padding) - (dilation * (kernel_size[1] - 1)) - 1)
            / stride
        )
        + 1
    )
    return h, w


def encoder_net(
    in_channels, input_size, hidden_channels: List[int], output_dim
):
    layers = []
    in_dim, in_size = in_channels, input_size
    # TODO question about the sequence of batch norm and activation
    for hidden_dim in hidden_channels:
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.LeakyReLU(),
            )
        )
        in_dim = hidden_dim
        in_size = conv2d_output_size(
            in_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
    layers.append(nn.Flatten(start_dim=1))
    preflattern_shape = (in_dim,) + in_size
    layers.append(
        nn.Linear(torch.prod(torch.tensor(preflattern_shape)), output_dim)
    )

    return nn.Sequential(*layers), preflattern_shape


def decoder_net(
    input_dim, conv_input_shape: Tuple[int], hidden_channels: List[int]
):
    layers = []
    in_dim = hidden_channels[0]
    layers.append(
        nn.Linear(input_dim, torch.prod(torch.tensor(conv_input_shape)))
    )
    layers.append(torch.nn.Unflatten(1, conv_input_shape))
    for hidden_dim in hidden_channels:
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.LeakyReLU(),
            )
        )
        in_dim = hidden_dim
    return nn.Sequential(*layers)


class AE(BaseAE):
    def __init__(
        self,
        in_channels: int,
        in_size: Tuple[int],
        hidden_channels: List[int],
        latent_dim: int,
        **kwargs
    ) -> None:
        super(AE, self).__init__()
        self.latent_dim = latent_dim
        conv_input_size = in_size

        self.encoder, preflattern_shape = encoder_net(
            in_channels,
            conv_input_size,
            hidden_channels,
            output_dim=latent_dim,
        )

        self.decoder = decoder_net(
            input_dim=latent_dim,
            conv_input_shape=preflattern_shape,
            hidden_channels=hidden_channels[::-1],
        )

        self.decoder_fc = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels[0],
                hidden_channels[0],
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.LeakyReLU(),
            nn.Conv2d(
                hidden_channels[0], out_channels=3, kernel_size=3, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, input) -> List[Tensor]:
        encoder_out = self.encoder(input)
        decoder_out = self.decoder(encoder_out)
        output = self.decoder_fc(decoder_out)
        return output

    def loss_function(self, output, target, **kwargs) -> dict:
        batch_size = output.shape[0]
        reconstruction_loss = (
            F.mse_loss(output, target, reduction="sum") / batch_size
        )
        loss = reconstruction_loss
        return loss
