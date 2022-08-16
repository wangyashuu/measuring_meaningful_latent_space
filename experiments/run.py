from calendar import c
import os
import yaml
import argparse

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from attrdict import AttrDict


from create_from_config import (
    create_datasets,
    create_dataloader,
    create_model,
    create_optimizer,
    create_scheduler,
)


class StepModule(pl.LightningModule):
    def __init__(self, model, optimizer, scheduler):
        super(StepModule, self).__init__()
        # self.hold_graph = self.params['retain_first_backpass'] or False
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def training_step(self, batch, batch_idx):
        x, _ = batch
        out = self.model(x)
        loss = self.model.loss_function(out, x)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        out = self.model(x)
        loss = self.model.loss_function(out, x)
        return loss
        # self.logger.log_metrics({f"val_{key}": val.item() for key, val in val_loss.items()})

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = self.scheduler
        return [optimizer], [scheduler]


def run(config):
    seed_everything(config.seed, True)
    train_set, test_set = create_datasets(config.data)
    train_loader = create_dataloader(train_set, config.train)
    test_loader = create_dataloader(test_set, config.val)
    model = create_model(config.model)
    optimizer = create_optimizer(model, config.optimizer)
    scheduler = create_scheduler(optimizer, config.scheduler)
    step_module = StepModule(model, optimizer, scheduler)

    runner = Trainer(
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(
                save_top_k=2,
                dirpath=os.path.join(config.logging.save_dir, "checkpoints"),
                monitor="val_loss",
                save_last=True,
            ),
        ],
        strategy=DDPPlugin(find_unused_parameters=False),
        # **config["trainer_params"],
    )
    runner.fit(
        step_module,
        train_loader,
        test_loader,
        max_epochs=config.trainer.max_epochs,
    )
    print("runner:")


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
            "trainer": {"max_epochs": 100},
        }
    )
    config = AttrDict(yaml.safe_load(file))
    run(default_config + config)
