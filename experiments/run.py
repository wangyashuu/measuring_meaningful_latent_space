import os
import yaml
import argparse

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    Callback,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from attrdict import AttrDict


from create_from_config import (
    create_datasets,
    create_dataloader,
    create_model,
    create_optimizers,
    create_schedulers,
    create_metrics,
)


class StepModule(pl.LightningModule):
    def __init__(self, model, optimizers, schedulers):
        super().__init__()
        # self.hold_graph = self.params['retain_first_backpass'] or False
        self.model = model
        self.optimizers = optimizers
        self.schedulers = schedulers

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch
        out = self.model(x)
        loss = self.model.loss_function(x, out, optimizer_idx)
        ###
        # remind, it is not comfortable to use logger.log_metrics directly,
        # since model checkpoint check the monitered key.
        # see change in this commit https://github.com/Lightning-AI/lightning/pull/12418/files
        # in order to add to monitered keys (callback_metrics), you should call self.log
        # P.S. return log will not log for you, see https://github.com/Lightning-AI/lightning/issues/5081
        ###
        self.log_dict({"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        out = self.model(x)
        loss = self.model.loss_function(x, out)
        self.log_dict({"val_loss": loss})

    def configure_optimizers(self):
        optimizers = self.optimizers
        schedulers = self.schedulers
        return optimizers, schedulers


class MetricsCallback(Callback):
    def __init__(self, data_loader, metrics):
        super().__init__()
        self.data_loader = data_loader
        self.metrics = metrics

    def on_fit_end(self, trainer, pl_module):
        x, y = next(iter(self.data_loader))
        encoded = pl_module.model.encode(x)
        log_metrics = {k: fn(y, encoded) for k, fn in self.metrics.items()}
       #  pl_module.log_dict(log_metrics)


def run(config):
    seed_everything(config.seed, True)
    # wandb auto set project name based on git (not documented), see: https://github.com/wandb/wandb/blob/cce611e2e518951064833b80aee975fa139a85ee/wandb/cli/cli.py#L872
    wandb_logger = WandbLogger(
        save_dir=config.logging.save_dir,
        group=f"{config.model.name}",
    )

    train_set, test_set = create_datasets(config.data)
    train_loader = create_dataloader(train_set, config.train)
    test_loader = create_dataloader(test_set, config.val)
    model = create_model(config.model)
    optimizers = create_optimizers(model, config.optimizers)
    schedulers = create_schedulers(optimizers, config.schedulers)
    metrics = create_metrics(config.metrics)
    step_module = StepModule(model, optimizers, schedulers)

    runner = Trainer(
        logger=wandb_logger,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(
                save_top_k=2,
                dirpath=os.path.join(config.logging.save_dir, "checkpoints"),
                monitor="val_loss",
                save_last=True,
            ),
            # MetricsCallback(data_loader=test_loader, metrics=metrics)
        ],
        strategy=DDPPlugin(find_unused_parameters=False),
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
    default_config = AttrDict(
        {
            "train": {"num_workers": 4},
            "val": {"num_workers": 4},
            "trainer": {"max_epochs": 4},
            "metrics": {"includes": []},
        }
    )
    config = AttrDict(yaml.safe_load(file))
    run(default_config + config)
