from typing import List, Tuple

import torch
import torch.nn as nn


## define fully connected encoder and decoder


def fc_encoder(input_shape, hiddens):
    print("fc_encoder", input_shape, hiddens)
    layers = [nn.Flatten(start_dim=1)]
    n_in = torch.prod(torch.tensor(input_shape))
    for h in hiddens:
        layers.append(nn.Sequential(nn.Linear(n_in, h), nn.ReLU()))
        n_in = h
    return nn.Sequential(*layers)


def fc_decoder(hiddens, output_shape):
    print("fc_decoder", output_shape, hiddens)
    layers = []
    n_in = hiddens[0]
    for h in hiddens[1:]:
        layers.append(nn.Sequential(nn.Linear(n_in, h), nn.ReLU()))
        n_in = h
    layers.append(
        nn.Sequential(
            nn.Linear(n_in, torch.prod(torch.tensor(output_shape))),
            nn.Sigmoid(),
        )
    )
    layers.append(nn.Unflatten(1, output_shape))
    return nn.Sequential(*layers)


## define convolutional neural network encoder and decoder


def conv2d_output_size(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    from math import floor

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    compute_output_length = (
        lambda l, kernel: (
            (l + (2 * padding) - (dilation * (kernel - 1)) - 1) / stride
        )
        + 1
    )

    h = compute_output_length(h_w[0], kernel_size[0])
    w = compute_output_length(h_w[1], kernel_size[1])
    output_padding_h = 1 if h > floor(h) else 0
    output_padding_w = 1 if w > floor(w) else 0

    return (floor(h), floor(w)), (output_padding_h, output_padding_w)


def cnn_encoder(input_shape: Tuple[int], hiddens: List[int]):
    print("cnn_encoder", input_shape, hiddens)
    output_paddings = []
    layers = []
    n_in, in_size = input_shape[0], input_shape[1:]
    # TODO question about the sequence of batch norm and activation
    for n_out in hiddens:
        layers.append(
            nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(n_out),
                nn.LeakyReLU(),
            )
        )
        n_in = n_out
        in_size, output_padding = conv2d_output_size(
            in_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        output_paddings.append(output_padding)
    output_shape = (n_in,) + in_size
    return nn.Sequential(*layers), output_shape, output_paddings


def cnn_decoder(hiddens: List[int], output_shape, output_paddings):
    print("cnn_decoder", output_paddings, hiddens)
    layers = []
    n_in = hiddens[0]
    for n_out, output_padding in zip(hiddens[1:], output_paddings):
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    n_in,
                    n_out,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=output_padding,
                ),
                nn.BatchNorm2d(n_out),
                nn.LeakyReLU(),
            )
        )
        n_in = n_out

    n_out = output_shape[0]
    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(
                n_in,
                n_out,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=output_paddings[-1],
            ),
            nn.Sigmoid(),
        )
    )
    return nn.Sequential(*layers)
