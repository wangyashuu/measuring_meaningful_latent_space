import os
import yaml
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    Callback,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy


from box import Box
import torchvision.utils as vutils
from torch.utils.data import DataLoader


from .create_from_config import (
    create_datasets,
    create_dataloader,
    create_model,
    create_optimizers,
    create_schedulers,
    create_metrics,
)


def get_data_from_dataset(dataset, size):
    loader = DataLoader(dataset, batch_size=size, shuffle=True)
    return next(iter(loader))


def get_data_from_dataloader(loader, size):
    xs, ys = [], []
    s = 0
    for x, y in loader:
        if s + x.shape[0] >= size:
            xs.append(x[: (size - s)])
            ys.append(y[: (size - s)])
            s += x.shape[0]
            break
        else:
            xs.append(x)
            ys.append(y)
            s += x.shape[0]
    return torch.vstack(xs), torch.vstack(ys)


class StepModule(pl.LightningModule):
    def __init__(self, model, optimizers, schedulers, metrics, test_data):
        super().__init__()
        # self.hold_graph = self.params['retain_first_backpass'] or False
        self.model = model
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.metrics = metrics
        self.test_data = test_data

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        x, _ = batch
        out = self.model(x)
        losses = self.model.loss_function(x, out, optimizer_idx)
        ###
        # remind, it is not comfortable to use logger.log_metrics directly,
        # since model checkpoint check the monitered key.
        # see change in this commit https://github.com/Lightning-AI/lightning/pull/12418/files
        # in order to add to monitered keys (callback_metrics), you should call self.log
        # P.S. return log will not log for you, see https://github.com/Lightning-AI/lightning/issues/5081
        ###
        self.log_dict({f"train/{k}": v for k, v in losses.items()})
        loss = losses.get("loss") or next(iter(losses.values()))
        return dict(loss=loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        losses = self.model.loss_function(x, out)
        y_hat = self.model.encode(x)
        self.log_dict(
            {f"val/{k}": v for k, v in losses.items()}, sync_dist=True
        )
        loss = losses.get("loss") or next(iter(losses.values()))
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        return dict(loss=loss, y_hat=y_hat, y=y)

    def validation_epoch_end(self, step_outs):
        # https://github.com/Lightning-AI/lightning/issues/13166#issuecomment-1196478740
        # torch.distributed.get_rank() or pl.utilities.rank_zero.rank_zero_only.rank
        y_hat = torch.vstack([o["y_hat"] for o in step_outs])
        y = torch.vstack([o["y"] for o in step_outs])
        log_metrics = {fn.__name__: fn(y, y_hat) for fn in self.metrics}
        self.log_dict(log_metrics, sync_dist=True)
        self.samples()
        self.reconstructs()

    def configure_optimizers(self):
        optimizers = self.optimizers
        schedulers = self.schedulers
        return optimizers, schedulers

    def samples(self):
        latents = torch.randn((144,) + self.model.latent_space).to(self.device)
        samples = self.model.decode(latents)
        image = vutils.make_grid(samples, normalize=False, nrow=12)
        self.logger.log_image(key="samples", images=[image])

    def reconstructs(self):
        x, _ = self.test_data
        x = x.to(self.device)
        result = self.model(x)
        reconstructions = result[0] if type(result) == tuple else result
        image = vutils.make_grid(reconstructions, normalize=False, nrow=12)
        self.logger.log_image(key="reconstructions", images=[image])
        x_image = vutils.make_grid(x, normalize=False, nrow=12)
        self.logger.log_image(key="originals", images=[x_image])


class MetricsCallback(Callback):
    def __init__(self, data_loader, metrics):
        super().__init__()
        self.data_loader = data_loader
        self.metrics = metrics


def run(config):
    seed_everything(config.seed, True)
    # wandb auto set project name based on git (not documented)
    # see: https://github.com/wandb/wandb/blob/cce611e2e518951064833b80aee975fa139a85ee/wandb/cli/cli.py#L872
    wandb_logger = WandbLogger(
        save_dir=config.logging.save_dir,
        group=f"{config.model.name}",
        config=config,
    )

    train_set, test_set = create_datasets(config.data)
    test_data = get_data_from_dataset(test_set, size=144)
    train_loader = create_dataloader(train_set, config.train)
    test_loader = create_dataloader(test_set, config.val)
    model = create_model(config.model)
    optimizers = create_optimizers(model, config.optimizers)
    schedulers = create_schedulers(optimizers, config.schedulers)
    metrics = create_metrics(config.metrics)
    step_module = StepModule(model, optimizers, schedulers, metrics, test_data)

    runner = Trainer(
        logger=wandb_logger,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(
                save_top_k=2,
                dirpath=os.path.join(config.logging.save_dir, "checkpoints"),
                monitor="val/loss",
                save_last=True,
            ),
        ],
        strategy=DDPStrategy(find_unused_parameters=False),
        # detect_anomaly=True,
        **config.trainer,
    )
    runner.fit(step_module, train_loader, test_loader)
    print("runner: end")


parser = argparse.ArgumentParser(description="Generic runner for VAE models")
parser.add_argument(
    "--config",
    "-c",
    dest="filename",
    metavar="FILE",
    help="path to the config file",
)

args = parser.parse_args()

with open(args.filename, "r") as file:
    default_config = Box(
        {
            "train": {"num_workers": 4},
            "val": {"num_workers": 4},
            "trainer": {"max_epochs": 4},
            "metrics": {"includes": []},
            "model": {"net_type": "cnn"},
        }
    )
    config = Box(yaml.safe_load(file))
    run(default_config + config)
