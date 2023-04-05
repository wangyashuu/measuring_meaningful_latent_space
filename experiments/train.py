import os
from pathlib import Path

import torch
from torch.utils.data import default_collate
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from lightning.fabric import seed_everything, Fabric
import wandb
from box import Box
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


# log({f"val/{k}": v for k, v in losses.items()}, step=step)
def log(data, columns=None, commit=False, step=None):
    if columns is not None:
        wrap = lambda x: wandb.Table(data=x, columns=columns)
    else:
        wrap = (
            lambda x: wandb.Image(x)
            if torch.is_tensor(x) and len(x.shape) > 1
            else x
        )
    data = {k: wrap(v) for k, v in data.items()}
    wandb.log(data, step=step, commit=commit)


class Trainer:
    def __init__(
        self,
        models,
        optimizers,
        device,
        schedulers=[],
        output_dir=".",
        n_steps_log_every=400,
        resume_path=None,
    ):
        self.models = [m.to(device) for m in models]
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.device = device
        self.output_dir = Path(output_dir)
        if resume_path is not None:
            self.resume(resume_path)

    def set_state(self, **kwargs):
        default_state = dict(train_size=0, val_size=0, step=0, epoch=0)
        self.state = Box({**(getattr(self, "state", default_state)), **kwargs})

    def resume(self, state_path):
        # torch.load issues
        # https://github.com/pytorch/pytorch/issues/2830#issuecomment-718816292
        for m in self.models:
            m.to(self.device)
        state_dicts, state = torch.load(state_path, map_location=self.device)
        for state_dict, m in zip(
            state_dicts, [*self.models, *self.optimizers, *self.schedulers]
        ):
            m.load_state_dict(state_dict)
        self.set_state(state)

    def _batch_to_device(self, batch):
        batch = (
            [b.to(self.device) for b in batch]
            if type(batch) is list
            else batch.to(self.device)
        )
        return batch

    def save(self, version="last"):
        state_dicts = [
            *[m.state_dict() for m in self.models],
            *[o.state_dict() for o in self.optimizers],
            *[s.state_dict() for s in self.schedulers],
        ]
        save_path = self.output_dir / "checkpoints"
        save_path.mkdir(exist_ok=True)
        torch.save((state_dicts), save_path / f"version={version}.ckpt")

    def log(self, log_dict, on_step=False, *args, **kwargs):
        if not on_step or (self.state.step + 1) % self.n_steps_log_every == 0:
            log(log_dict, *args, step=self.state.step, **kwargs)

    def _train_batch(self, batch, batch_idx, loss_calcs):
        step = self.state.epoch * self.state.train_size + batch_idx
        self.set_state(step=step)
        common_params = dict(step=step, dataset_size=self.state.train_size)
        for model, optimizer, loss_calc in zip(
            self.models, self.optimizers, loss_calcs
        ):
            model.train()
            optimizer.zero_grad()
            losses = loss_calc(batch, *self.models, **common_params)
            loss = losses.get("loss") or next(iter(losses.values()))
            loss.backward()
            optimizer.step()
            self.log(losses, on_step=True)
            model.eval()

    def _val_batch(self, batch, batch_idx, loss_calcs):
        params = dict(step=self.state.step, dataset_size=self.state.val_size)
        rs = [calc(batch, *self.models, **params) for calc in loss_calcs]
        return rs

    def val(self, val_loader, loss_calcs):
        self.set_state(val_size=len(val_loader.dataset))
        outs = []
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Val")):
            batch = self._batch_to_device(batch)
            outs.append(self._val_batch(batch, batch_idx, loss_calcs))
        return {k: o[k].mean() for o in default_collate(outs) for k in o}

    def test(self, test_loader, test_batch):
        outs = []
        for batch_idx, batch in enumerate(test_loader):
            batch = self._batch_to_device(batch)
            outs.append(test_batch(batch, batch_idx, *self.models))
        return default_collate(outs)

    def train(
        self,
        n_epochs,
        loss_calcs,
        train_loader,
        val_loader=None,
        test_loader=None,
        test_batch=None,
        on_test_end=None,
        n_epochs_test_every=1,
        n_epochs_save_every=1,
    ):
        self.set_state(train_size=len(train_loader.dataset))
        start_epoch = self.state.epoch
        for epoch in range(start_epoch, n_epochs):
            for batch_idx, batch in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch}")
            ):
                batch = self._batch_to_device(batch)
                self._train_batch(batch, batch_idx, loss_calcs)
            for scheduler in self.schedulers:
                scheduler.step()
            with torch.no_grad():
                if val_loader is not None:
                    losses = self.val(val_loader, loss_calcs)
                    self.log(losses)
                if (
                    test_loader is not None
                    and (epoch + 1) % n_epochs_test_every == 0
                ):
                    test_rs = self.test(test_loader, test_batch)
                    if on_test_end is not None:
                        on_test_end(test_rs, self)

            if (epoch + 1) % n_epochs_save_every == 0:
                self.save(version=epoch)
            self.log(self.state, commit=True)
            self.set_state(epoch=epoch)


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


def samples(vae, device, size=144):
    latents = torch.randn((size,) + vae.latent_space, device=device)
    samples = vae.decode(latents)
    image = vutils.make_grid(F.sigmoid(samples), normalize=False, nrow=12)
    return dict(samples=image)


def reconstructs(vae, dataset, device, size=144):
    x, _ = next(iter(DataLoader(dataset, batch_size=size, shuffle=True)))
    x = x.to(device)
    result = vae(x)
    reconstructions = result[0] if type(result) == tuple else result
    image = vutils.make_grid(
        F.sigmoid(reconstructions), normalize=False, nrow=12
    )
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
    output_dir = Path(config.output_dir) / config.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config.seed)
    train_set, val_set = create_datasets(config.data)
    train_loader = DataLoader(
        train_set, pin_memory=True, **config.dataloader.train
    )
    val_loader = DataLoader(val_set, pin_memory=True, **config.dataloader.val)
    models, optimizers, schedulers, loss_calcs = init(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_funcs = create_metrics(config.metrics)
    run = wandb.init(
        project="innvariant-representations",
        group=config.model_name,
        dir=output_dir,
        config=config,
    )

    def test_batch(batch, batch_idx, vae, *args, **kwargs):
        x, y = batch
        y_hat = vae.encode(x)
        return y, y_hat

    def on_test_end(collated, trainer):
        [y, y_hat] = collated
        y = y.reshape(-1, *y.shape[2:]).cpu().numpy()
        y_hat = y_hat.reshape(-1, *y_hat.shape[2:]).cpu().numpy()
        metrics = run_metrics(y, y_hat, metric_funcs)
        vae = trainer.models[0]
        images = {
            **samples(vae, device=trainer.device),
            **reconstructs(vae, val_loader.dataset, device=trainer.device),
        }
        return {**metrics, **images}

    trainer = Trainer(
        models,
        optimizers,
        schedulers=schedulers,
        device=device,
        output_dir=output_dir,
    )
    trainer.train(
        n_epochs=config.n_epochs,
        loss_calcs=loss_calcs,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=val_loader,
        test_batch=test_batch,
        on_test_end=on_test_end,
        n_epochs_test_every=config.test_every_k_epoch,
    )
