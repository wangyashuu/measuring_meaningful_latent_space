import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from lightning.fabric import seed_everything, Fabric
import wandb
from tqdm import tqdm

from disentangling.metrics import get_mutual_infos
from .create_from_config import (
    create_datasets,
    create_model,
    create_auxiliary_model,
    create_compute_loss,
    create_optimizer,
    create_scheduler,
    create_metrics,
)


def init(config):
    train_set, val_set = create_datasets(config.data)
    train_loader = DataLoader(
        train_set, pin_memory=True, **config.dataloader.train
    )
    val_loader = DataLoader(val_set, pin_memory=True, **config.dataloader.val)

    model_name = config.model_name
    main_model = create_model(
        model_name=model_name,
        input_shape=config.data.input_shape,
        net_params=config.net_params,
        latent_dim=config.latent_dim,
    )
    compute_model_loss = create_compute_loss(
        model_name, config.get("loss_kwargs", {})
    )
    optimizer = create_optimizer(main_model, config.optimizer)
    model_names = [model_name]
    models = [main_model]
    compute_loss_functions = [compute_model_loss]
    optimizers = [optimizer]
    schedulers = []
    if "scheduler" in config:
        schedulers.append(create_scheduler(optimizer, config.scheduler))

    if "auxiliary" in config:
        aux_conf = config.auxiliary
        aux_name = aux_conf.model.name
        aux_model = create_auxiliary_model(
            aux_conf.model, latent_dim=config.get("latent_dim", None)
        )
        compute_aux_loss = create_compute_loss(
            aux_name, aux_conf.get("loss_kwargs", {})
        )
        aux_optimizer = create_optimizer(aux_model, aux_conf.optimizer)
        model_names.append(aux_name)
        models.append(aux_model)
        compute_loss_functions.append(compute_aux_loss)
        optimizers.append(aux_optimizer)
        if "scheduler" in aux_conf:
            aux_scheduler = create_scheduler(aux_optimizer, aux_conf.scheduler)
            schedulers.append(aux_scheduler)

    return (
        train_loader,
        val_loader,
        model_names,
        models,
        optimizers,
        schedulers,
        compute_loss_functions,
    )


def samples(vae, size=144):
    latents = torch.randn((size,) + vae.latent_space, device=vae.device)
    samples = vae.decode(latents)
    image = vutils.make_grid(F.sigmoid(samples), normalize=False, nrow=12)
    return dict(samples=image)


def reconstructs(vae, dataset, size=144):
    x, _ = next(iter(DataLoader(dataset, batch_size=size, shuffle=True)))
    x = x.to(vae.device)
    result = vae(x)
    reconstructions = result[0] if type(result) == tuple else result
    image = vutils.make_grid(F.sigmoid(reconstructions), normalize=False, nrow=12)
    x_image = vutils.make_grid(x, normalize=False, nrow=12)
    return dict(reconstructions=image, originals=x_image)


def run_metrics(y, y_hat, metric_funcs):
    all_results = {}
    for fn in metric_funcs:
        res = fn(y, y_hat)
        all_results.update(
            {f"{fn.__name__}__{k}": res[k] for k in res}
            if type(res) is dict
            else {fn.__name__: res}
        )
    return all_results


def check_correlation(y, y_hat):
    columns = list(range(y.shape[1]))
    matrices = dict(
        mi_matrix=get_mutual_infos(y_hat, y, estimator="ksg", normalized=True)
    )
    return matrices, columns


def train(config):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    seed_everything(config.seed)
    (
        train_loader,
        val_loader,
        model_names,
        models,
        optimizers,
        schedulers,
        compute_loss_functions,
    ) = init(config)
    fabric = Fabric(**config.fabric, strategy="ddp")
    fabric.launch()

    output_dir = Path(config.output_dir) / config.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    from_n_epoch = 0
    if "saved_state" in config:
        # torch.load issues
        # https://github.com/pytorch/pytorch/issues/2830#issuecomment-718816292
        for m in models:
            m.to(fabric.device)
        state_dicts, epoch = torch.load(
            config.saved_state, map_location=fabric.device
        )
        for state_dict, m in zip(
            state_dicts, [*models, *optimizers, *schedulers]
        ):
            m.load_state_dict(state_dict)
        from_n_epoch = epoch + 1
        del state_dicts, epoch

    train_loader = fabric.setup_dataloaders(train_loader)
    val_loader = fabric.setup_dataloaders(val_loader)
    models, optimizers = list(
        zip(*[fabric.setup(m, o) for m, o in zip(models, optimizers)])
    )

    if fabric.is_global_zero:
        run = wandb.init(
            project="innvariant-representations",
            group=config.model_name,
            dir=output_dir,
            config=config,
        )

    def log(data, columns=None, commit=False, step=None):
        if fabric.is_global_zero:
            if columns is not None:
                wrap = lambda x: wandb.Table(data=x, columns=columns)
            else:
                wrap = (
                    lambda x: wandb.Image(x)
                    if torch.is_tensor(x) and len(x.shape) > 1
                    else x
                )
            data = {k: wrap(v) for k, v in data.items()}
            if step > run.step:
                run.log({}, step=run.step, commit=True)
            run.log(data, step=step, commit=commit)

    model_dict = {name: m for name, m in zip(model_names, models)}
    metric_funcs = create_metrics(config.metrics)

    def train_step(batch, batch_idx, step):
        for model, optimizer, compute_loss in zip(
            models, optimizers, compute_loss_functions
        ):
            model.train()
            optimizer.zero_grad()
            losses = compute_loss(
                batch[0],
                **model_dict,
                step=step,
                dataset_size=len(train_loader.dataset),
            )
            loss = losses.get("loss") or next(iter(losses.values()))
            fabric.backward(loss)
            optimizer.step()
            if (step + 1) % config.log_every_k_step == 0:
                log({f"train/{k}": v for k, v in losses.items()}, step=step)
            model.eval()

    def val_step(batch, batch_idx, step):
        for model, compute_loss in zip(models, compute_loss_functions):
            losses = compute_loss(
                batch[0],
                **model_dict,
                step=step,
                dataset_size=len(val_loader.dataset),
            )
            log({f"val/{k}": v for k, v in losses.items()}, step=step)

    def test_step(batch, batch_idx, step):
        vae = models[0]
        x, y = batch
        y_hat = vae.encode(x)
        return y, y_hat

    def after_test(outs, epoch, step):
        if "latent_dim" in config:
            images = {
                **samples(vae=models[0]),
                **reconstructs(vae=models[0], dataset=val_loader.dataset),
            }
            log(images, step=step)
            Y, Y_hat = [torch.vstack(out).cpu().numpy() for out in zip(*outs)]
            log(run_metrics(Y, Y_hat, metric_funcs), step=step)
            # log(*check_correlation(Y, Y_hat), step=step)

    def start_epoch(epoch, step):
        lrs = {
            f"{name}_learning_rate": scheduler.get_last_lr()[0]
            for name, scheduler in zip(model_names, schedulers)
        }
        log({**lrs, "epoch": epoch}, step=step)

    def end_epoch(epoch, step):
        if fabric.is_global_zero:
            state_dicts = [
                *[m.state_dict() for m in models],
                *[o.state_dict() for o in optimizers],
                *[s.state_dict() for s in schedulers],
            ]
            save_path = output_dir / "checkpoints"
            save_path.mkdir(exist_ok=True)
            torch.save((state_dicts, epoch), save_path / f"epoch={epoch}.ckpt")
        log({}, step=step, commit=True)

    for epoch in range(from_n_epoch, config.n_epochs):
        step = (epoch - from_n_epoch) * len(train_loader)
        start_epoch(epoch, step)
        for batch_idx, batch in tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Train Epoch {epoch}",
            disable=not fabric.is_global_zero,
        ):
            step = (epoch - from_n_epoch) * len(train_loader) + batch_idx
            train_step(batch, batch_idx, step)

        for scheduler in schedulers:
            scheduler.step()

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(val_loader),
                desc=f"Val Epoch {epoch}",
                disable=not fabric.is_global_zero,
            ):
                val_step(batch, batch_idx, step)

            if (epoch + 1) % config.test_every_k_epoch == 0:
                outs = []
                for batch_idx, batch in enumerate(val_loader):
                    outs.append(test_step(batch, batch_idx, step))
                after_test(outs, epoch, step)
                outs.clear()
        end_epoch(epoch, step)
