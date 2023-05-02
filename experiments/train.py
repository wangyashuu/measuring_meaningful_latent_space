from pathlib import Path
import urllib.parse

import torch
from torch.utils.data import DataLoader
from .utils.solver import Solver
from .utils.logger import WandbLogger
from .utils.ddp import ddp_dataloader, ddp_model, ddp_run
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


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def init_loss(model, config):
    loss_calc = create_compute_loss(
        config.model_name, config.get("loss_kwargs", {})
    )
    optimizer = create_optimizer(model, config.optimizer)
    scheduler = None
    if "scheduler" in config:
        scheduler = create_scheduler(optimizer, config.scheduler)

    return loss_calc, optimizer, scheduler


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
    seed_everything(config.seed)
    name = urllib.parse.urlencode(config.loss_kwargs) or "empty"
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
    discrete_factors = val_set.dataset.discrete_factors
    if rank >= 0:
        train_loader = ddp_dataloader(train_loader)
        val_loader = ddp_dataloader(val_loader)
        models = [ddp_model(m, rank) for m in models]

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
        if trainer._is_ddp():
            vae = vae.module
        images = {
            **samples(vae, device=trainer.device),
            **reconstructs(vae, val_loader.dataset, device=trainer.device),
        }
        # trainer.log(*check_correlation(y, y_hat))
        return {**images, **metrics}

    logger = WandbLogger(
        is_on_master=rank < 1, n_steps_log_every=config.n_steps_log_every
    )
    logger.start(
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
        logger=logger,
        ddp_rank=rank,
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


def train(ddp, config):
    if ddp:
        ddp_run(run_train, config)
    else:
        run_train(-1, config)
