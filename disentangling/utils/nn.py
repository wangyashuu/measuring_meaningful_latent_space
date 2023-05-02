from typing import List, Tuple

import torch
import torch.nn as nn


## define fully connected encoder and decoder
# TODO: note https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout

conv_params = dict(kernel_size=4, stride=2, padding=1)


def get_layer_modules(layer, batch_norm=False, activation=True):
    modules = [layer]
    if batch_norm:
        normed = (
            nn.BatchNorm1d(layer.out_features)
            if isinstance(layer, nn.Linear)
            else nn.BatchNorm2d(layer.out_channels)
        )
        modules.append(normed)
    if activation:
        # if activation == "ReLU":
        #     activated = nn.ReLU()
        # else:
        #     activated = nn.LeakyReLU()
        modules.append(nn.ReLU())
    if len(modules) > 1:
        modules = [nn.Sequential(*modules)]
    return modules


def get_fc_encoder(input_shape, hiddens, batch_norms=False, activations=True):
    if type(batch_norms) is not list:
        batch_norms = [batch_norms] * len(hiddens)
    if type(activations) is not list:
        activations = [activations] * len(hiddens)
    layers = [nn.Flatten(start_dim=1)]
    n_in = torch.prod(torch.tensor(input_shape))
    for n_out, batch_norm, activation in zip(
        hiddens, batch_norms, activations
    ):
        layer = nn.Linear(n_in, n_out)
        layers.extend(get_layer_modules(layer, batch_norm, activation))
        n_in = n_out
    return nn.Sequential(*layers)


def get_fc_decoder(
    hiddens, output_shape, batch_norms=False, activations=True, is_output=True
):
    if type(batch_norms) is not list:
        batch_norms = [batch_norms] * len(hiddens)
    if type(activations) is not list:
        activations = [activations] * len(hiddens)
    layers = []
    n_in = hiddens[0]
    unflatten_input_dims = torch.prod(torch.tensor(output_shape))
    n_outs = hiddens[1:] if is_output else hiddens[1:] + [unflatten_input_dims]
    for n_out, batch_norm, activation in zip(n_outs, batch_norms, activations):
        layer = nn.Linear(n_in, n_out)
        layers.extend(get_layer_modules(layer, batch_norm, activation))
        n_in = n_out

    if is_output:
        layers.append(nn.Linear(n_in, unflatten_input_dims))

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


def get_cnn_encoder(
    input_shape: Tuple[int],
    hiddens: List[int],
    batch_norms=False,
    activations=True,
):
    if type(batch_norms) is not list:
        batch_norms = [batch_norms] * len(hiddens)
    if type(activations) is not list:
        activations = [activations] * len(hiddens)
    output_paddings = []
    layers = []
    n_in, in_size = input_shape[0], input_shape[1:]
    # TODO question about the sequence of batch norm and activation
    for n_out, batch_norm, activation in zip(
        hiddens, batch_norms, activations
    ):
        layer = nn.Conv2d(n_in, n_out, **conv_params)
        layers.extend(get_layer_modules(layer, batch_norm, activation))
        n_in = n_out
        in_size, output_padding = conv2d_output_size(in_size, **conv_params)
        output_paddings.append(output_padding)
    output_shape = (n_in,) + in_size
    return nn.Sequential(*layers), output_shape, output_paddings


def get_cnn_decoder(
    hiddens: List[int],
    output_shape,
    output_paddings,
    batch_norms=False,
    activations=True,
):
    if type(batch_norms) is not list:
        batch_norms = [batch_norms] * len(hiddens)
    if type(activations) is not list:
        activations = [activations] * len(hiddens)

    layers = []
    n_in = hiddens[0]
    # TODO: check why sigmoid not work for not output network.
    for n_out, output_padding, batch_norm, activation in zip(
        hiddens[1:], output_paddings, batch_norms, activations
    ):
        layer = nn.ConvTranspose2d(
            n_in, n_out, output_padding=output_padding, **conv_params
        )
        layers.extend(get_layer_modules(layer, batch_norm, activation))
        n_in = n_out
    n_out = output_shape[0]
    layers.append(
        nn.ConvTranspose2d(
            n_in, n_out, output_padding=output_paddings[-1], **conv_params
        ),
    )
    return nn.Sequential(*layers)
