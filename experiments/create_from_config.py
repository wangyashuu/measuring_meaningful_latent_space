import torch
from torch.utils.data import DataLoader

from disentangling.datasets import CelebA_sets
from disentangling.models import AE, VAE, BetaVAE


def create_datasets(conf):
    if conf.name == "CelebA":
        return CelebA_sets()


def create_dataloader(dataset, conf):
    return DataLoader(
        dataset,
        batch_size=conf.batch_size,
        num_workers=conf.num_workers,
        # shuffle=True,
        # pin_memory=conf.pin_memory,
    )


def create_model(conf):
    if conf.name == "AE":
        return AE(conf.input_shape, conf.hidden_channels, conf.latent_dim)
    elif conf.name == "VAE":
        return VAE(conf.input_shape, conf.hidden_channels, conf.latent_dim)
    elif conf.name == "BetaVAE":
        return BetaVAE(
            conf.input_shape, conf.hidden_channels, conf.latent_dim, conf.beta
        )
    raise Exception(f"model {conf.name} not implemented")


def create_optimizer(model, conf):
    return torch.optim.Adam(
        model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay
    )


def create_scheduler(optimizer, conf):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=conf.gamma)
