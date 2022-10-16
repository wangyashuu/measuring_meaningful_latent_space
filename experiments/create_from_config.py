import torch
from torch.utils.data import DataLoader

import disentangling.datasets
import disentangling.models
import disentangling.metrics

from disentangling.utils.nn import (
    fc_encoder,
    fc_decoder,
    cnn_decoder,
    cnn_encoder,
)


def create_datasets(conf):
    name = conf.pop("name")
    D = getattr(disentangling.datasets, name)
    if not hasattr(disentangling.datasets, name):
        raise Exception(f"dataset {name} not implemented")
    return D(**conf)


def create_dataloader(dataset, conf):
    return DataLoader(
        dataset,
        batch_size=conf.batch_size,
        num_workers=conf.num_workers,
        shuffle=conf.pop('shuffle', False),
        # pin_memory=conf.pin_memory,
    )


def create_model(conf):
    name = conf.pop("name")
    net_type = conf.pop("net_type")
    input_shape = conf.pop("input_shape")
    hiddens = conf.pop("hiddens")
    if net_type == "fc":
        encoder = fc_encoder(input_shape=input_shape, hiddens=hiddens)
        decoder = fc_decoder(hiddens=hiddens[::-1], output_shape=input_shape)
        encoder_output_shape = (hiddens[-1],)
    else:
        encoder, encoder_output_shape, output_paddings = cnn_encoder(
            input_shape=input_shape,
            hiddens=hiddens,
        )
        decoder = cnn_decoder(
            hiddens=hiddens[::-1],
            output_paddings=output_paddings[::-1],
            output_shape=input_shape,
        )

    if not hasattr(disentangling.models, name):
        raise Exception(f"model {name} not implemented")

    model = getattr(disentangling.models, name)
    return model(
        encoder=encoder,
        decoder=decoder,
        encoder_output_shape=encoder_output_shape,
        decoder_input_shape=encoder_output_shape,
        **conf,
    )


def create_optimizers(model, confs):
    optimizers = []
    for c in confs:
        m = getattr(model, c.model) if "model" in c else model
        o = torch.optim.Adam(
            m.parameters(), lr=c.lr, weight_decay=c.weight_decay
        )
        optimizers.append(o)
    return optimizers


def create_schedulers(optimizers, confs):
    return [
        torch.optim.lr_scheduler.ExponentialLR(o, gamma=c.gamma)
        for o, c in zip(optimizers, confs)
    ]


def create_metrics(conf):
    return [
        getattr(disentangling.metrics, metric_name)
        for metric_name in conf.includes
    ]
