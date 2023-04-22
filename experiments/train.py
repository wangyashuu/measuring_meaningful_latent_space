from pathlib import Path
import urllib.parse

import torch
from torch.utils.data import DataLoader
from .utils.solver import Solver
from .utils.logger import WandbLogger
from .utils.ddp import ddp_dataloader, ddp_run
from .test import samples, reconstructs, run_metrics, test_batch

from .utils.create_from_config import (
    create_datasets,
    create_model,
    create_auxiliary_model,
    create_compute_loss,
    create_optimizer,
    create_scheduler,
    create_metrics,
)


def init_main(config):
    main_model = create_model(
        model_name=config.model_name,
        input_shape=config.data.input_shape,
        net_params=config.net_params,
        latent_dim=config.latent_dim,
    )
    loss_calc = create_compute_loss(
        config.model_name, config.get("loss_kwargs", {})
    )
    optimizer = create_optimizer(main_model, config.optimizer)
    scheduler = None
    if "scheduler" in config:
        scheduler = create_scheduler(optimizer, config.scheduler)
    return main_model, optimizer, scheduler, loss_calc


def init_auxiliary(config):
    aux_conf = config.auxiliary
    aux_name = aux_conf.model.name
    aux_model = create_auxiliary_model(
        aux_conf.model, latent_dim=config.get("latent_dim", None)
    )
    loss_calc = create_compute_loss(aux_name, aux_conf.get("loss_kwargs", {}))
    aux_optimizer = create_optimizer(aux_model, aux_conf.optimizer)
    aux_scheduler = None
    if "scheduler" in aux_conf:
        aux_scheduler = create_scheduler(aux_optimizer, aux_conf.scheduler)
    return aux_model, aux_optimizer, aux_scheduler, loss_calc


def init(config):
    rs = init_main(config)
    return_values = (
        zip(rs, init_auxiliary(config)) if "auxiliary" in config else zip(rs)
    )
    return [[x for x in pairs if x is not None] for pairs in return_values]


def run_train(rank, config):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = rank

    name = urllib.parse.urlencode(config.loss_kwargs)
    output_dir = (
        Path(config.output_dir) / config.model_name / str(config.seed) / name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    models, optimizers, schedulers, loss_calcs = init(config)
    train_set, val_set = create_datasets(config.data)
    train_loader = DataLoader(
        train_set, **config.dataloader.train, pin_memory=True
    )
    val_loader = DataLoader(val_set, **config.dataloader.val, pin_memory=True)
    train_loader = ddp_dataloader(train_loader)
    val_loader = ddp_dataloader(val_loader)
    discrete_factors = val_set.dataset.discrete_factors

    metric_funcs = create_metrics(config.metrics)

    def on_test_end(collated, trainer):
        y = collated[0].cpu().numpy()
        y_hat = collated[1].cpu().numpy()
        metrics = run_metrics(
            y,
            y_hat,
            metric_funcs,
            discrete_factors=discrete_factors,
        )
        vae = trainer.models[0]
        images = {
            **samples(vae, device=trainer.device),
            **reconstructs(vae, val_loader.dataset, device=trainer.device),
        }
        # trainer.log(*check_correlation(y, y_hat))
        return {**images, **metrics}

    logger = WandbLogger(
        device=0,
        project="innvariant-representations",
        group=config.model_name,
        name=name,
        dir=output_dir,
        config=config,
    )
    trainer = Solver(
        models,
        optimizers=optimizers,
        schedulers=schedulers,
        device=device,
        logger=logger,
    )
    trainer.train(
        n_epochs=config.n_epochs,
        loss_calcs=loss_calcs,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=val_loader,
        test_batch=test_batch,
        on_test_end=on_test_end,
        output_dir=output_dir,
        n_epochs_test_every=config.n_epochs_test_every,
        n_epochs_save_every=config.n_epochs_save_every,
    )


def train(config):
    ddp_run(run_train, config)
