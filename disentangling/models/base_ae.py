from typing import List, Tuple
from abc import abstractmethod

import torch
from torch import nn, Tensor


def conv2d_output_size(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    from math import floor

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    compute_output_length = lambda l, kernel: floor(
        ((l + (2 * padding) - (dilation * (kernel - 1)) - 1) / stride) + 1
    )

    h = compute_output_length(h_w[0], kernel_size[0])
    w = compute_output_length(h_w[1], kernel_size[1])
    return h, w


def get_encoder_net(input_shape: Tuple[int], hidden_channels: List[int]):
    layers = []
    n_in_channels, in_size = input_shape[0], input_shape[1:]
    # TODO question about the sequence of batch norm and activation
    for n_out_channels in hidden_channels:
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    n_in_channels,
                    n_out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(n_out_channels),
                nn.LeakyReLU(),
            )
        )
        n_in_channels = n_out_channels
        in_size = conv2d_output_size(
            in_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
    output_shape = (n_in_channels,) + in_size
    return nn.Sequential(*layers), output_shape


def get_encoder_fc(input_shape, output_dim):
    return nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(torch.prod(torch.tensor(input_shape)), output_dim),
    )


def get_decoder_fc(input_dim, output_shape):
    return nn.Sequential(
        nn.Linear(input_dim, torch.prod(torch.tensor(output_shape))),
        nn.Unflatten(1, output_shape),
    )


def get_decoder_net(input_shape: Tuple[int], hidden_channels: List[int]):
    layers = []
    n_in_channels, in_size = input_shape[0], input_shape[1:]
    for n_out_channels in hidden_channels:
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    n_in_channels,
                    n_out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(n_out_channels),
                nn.LeakyReLU(),
            )
        )
        n_in_channels = n_out_channels

    layers.append(
        nn.Sequential(
            nn.Conv2d(
                n_in_channels, n_in_channels, kernel_size=3, padding="same"
            ),
            nn.Tanh(),
        )
    )
    return nn.Sequential(*layers)


class BaseAE(nn.Module):
    def __init__(
        self,
        input_shape,
        hidden_channels,
    ) -> None:
        super().__init__()
        encoder_net, encoded_shape = get_encoder_net(
            input_shape, hidden_channels
        )
        decoder_net = get_decoder_net(
            input_shape=encoded_shape,
            hidden_channels=hidden_channels[::-1] + [input_shape[0]],
        )
        self.encoder_net = encoder_net
        self.encoded_shape = encoded_shape
        self.decoder_net = decoder_net

    def encode(self, input: Tensor) -> Tensor:
        encoded = self.encoder_net(input)
        return encoded

    def decode(self, input: Tensor) -> Tensor:
        decoded = self.decoder_net(input)
        return decoded

    def forward(self, input: Tensor) -> Tensor:
        encoded = self.encoder_net(input)
        decoded = self.decoder_net(encoded)
        return decoded

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def loss_function(self, *inputs: Tensor, **kwargs) -> Tensor:
        pass
