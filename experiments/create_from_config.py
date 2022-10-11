import torch
from torch.utils.data import DataLoader

from disentangling.datasets import CelebA_sets, dSprites_sets
import disentangling.models

# from disentangling.models import AE, VAE, BetaVAE, FactorVAE, BetaTCVAE
import disentangling.metrics


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
    name = conf.pop("name")
    if not hasattr(disentangling.models, name):
        raise Exception(f"model {name} not implemented")

    model = getattr(disentangling.models, name)
    return model(**conf)


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
