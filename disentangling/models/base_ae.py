from typing import List, Tuple
from abc import abstractmethod

import torch
from torch import nn, Tensor


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
    input_shape: Tuple[int],
    hidden_channels: List[int],
    output_dim,
):
    layers = []
    in_dim, in_size = input_shape[0], input_shape[1:]
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
    preflattern_size = in_size
    layers.append(
        nn.Linear(
            in_dim * torch.prod(torch.tensor(preflattern_size)), output_dim
        )
    )
    return nn.Sequential(*layers), preflattern_size


def decoder_net(
    input_dim: int, hidden_channels: List[int], conv_input_size: Tuple[int]
):
    layers = []
    in_dim = hidden_channels[0]
    layers.append(
        nn.Linear(input_dim, in_dim*torch.prod(torch.tensor(conv_input_size)))
    )
    layers.append(torch.nn.Unflatten(1, (in_dim,) + conv_input_size))
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
    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.LeakyReLU(),
            nn.Conv2d(in_dim, out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )
    )
    return nn.Sequential(*layers)


class BaseAE(nn.Module):
    def __init__(
        self,
        encoder_input_shape,
        encoder_output_dim,
        decoder_input_dim,
        hidden_channels,
    ) -> None:
        super().__init__()
        self.encoder, preflattern_size = encoder_net(
            input_shape=encoder_input_shape,
            hidden_channels=hidden_channels,
            output_dim=encoder_output_dim,
        )
        self.decoder = decoder_net(
            input_dim=decoder_input_dim,
            conv_input_size=preflattern_size,
            hidden_channels=hidden_channels[::-1],
            # output_shape=encoder_input_shape,
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def loss_function(self, *inputs: Tensor, **kwargs) -> Tensor:
        pass
