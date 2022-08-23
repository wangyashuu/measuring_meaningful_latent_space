import torch
from torch.utils.data import DataLoader

from disentangling.datasets import CelebA_sets, dSprites_sets
from disentangling.models import AE, VAE, BetaVAE
from disentangling.metrics import mig, mig_sup
from disentangling.models.factor_vae import FactorVAE


def create_datasets(conf):
    if conf.name == "CelebA":
        return CelebA_sets()
    elif conf.name == "dSprites":
        return dSprites_sets()
    raise Exception(f"dataset {conf.name} not implemented")


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
    elif conf.name == "FactorVAE":
        return FactorVAE(
            conf.input_shape, conf.hidden_channels, conf.latent_dim, conf.gamma
        )
    raise Exception(f"model {conf.name} not implemented")


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
    if "mig" in conf.includes:
        return {"mig": mig, "mig_sup": mig_sup}
    return {}
