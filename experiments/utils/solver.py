from pathlib import Path

import torch
from torch.utils.data import default_collate
from box import Box
from tqdm import tqdm
import torch.distributed as dist

from .ddp import ddp_model, ddp_dataloader


def is_time_to_do(n, n_do_every):
    return n_do_every > 0 and ((n + 1) % n_do_every == 0)


def collate(result, collect=None):
    collect = collect or (lambda x: torch.vstack(x))
    item = result[0]
    item_type = type(item)
    if item_type is tuple:
        return tuple(
            [collect([r[i] for r in result]) for i in range(len(item))]
        )
    elif item_type is dict:
        return {k: collect([r[k] for r in result]) for k in item}
    else:
        return collect(result)


class Solver:
    def __init__(
        self,
        models,
        optimizers=[],
        schedulers=[],
        logger=None,
        resume_path=None,
        ddp_rank=-1,
    ):
        is_ddp = ddp_rank >= 0
        if is_ddp:
            models = [ddp_model(m, ddp_rank) for m in models]

        self._models = models
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.logger = logger
        if resume_path is not None:
            self.resume(resume_path)
        self.ddp_rank = ddp_rank
        self.device = ddp_rank
        if self.ddp_rank < 0:
            self.device = torch.cuda.current_device()
            for m in self.models:
                m.to(self.device)

    @property
    def models(self):
        if self.is_ddp:
            return [getattr(m, "module", m) for m in self._models]
        return self._models

    @property
    def is_ddp(self):
        return self.ddp_rank >= 0

    def set_state(self, **kwargs):
        default_state = dict(train_size=0, val_size=0, step=0, epoch=0)
        self.state = Box({**(getattr(self, "state", default_state)), **kwargs})

    def resume(self, state_path):
        # torch.load issues
        # https://github.com/pytorch/pytorch/issues/2830#issuecomment-718816292
        for m in self._models:
            m.to(self.device)
        state_dicts, state = torch.load(state_path)

        components = self.models + self.optimizers + self.schedulers
        for state_dict, c in zip(state_dicts, components):
            c.load_state_dict(state_dict)
        self.set_state(epoch=state)

    def _batch_to_device(self, batch):
        batch = (
            [b.to(self.device) for b in batch]
            if type(batch) is list
            else batch.to(self.device)
        )
        return batch

    def save(self, output_dir=".", version="last"):
        if (not self.is_ddp) or self.device == 0:
            components = self.models + self.optimizers + self.schedulers
            state_dicts = [c.state_dict() for c in components]
            save_path = Path(output_dir) / "checkpoints"
            save_path.mkdir(exist_ok=True)
            save_state = (state_dicts, self.state.epoch)
            torch.save(save_state, save_path / f"version={version}.ckpt")

    def log(self, data, columns=None, step=None, **kwargs):
        step = step or self.state.step
        self.logger.log(data, columns=columns, step=step, **kwargs)

    def _train_batch(self, batch, batch_idx, loss_calcs):
        step = self.state.epoch * self.state.n_train_batchs + batch_idx
        self.set_state(step=step)
        common_params = dict(step=step, dataset_size=self.state.train_size)
        for model, optimizer, loss_calc in zip(
            self.models, self.optimizers, loss_calcs
        ):
            model.train()
            optimizer.zero_grad()
            losses = loss_calc(batch, *self.models, **common_params)
            loss = losses.get("loss") or next(iter(losses.values()))
            loss.backward()  # get grad
            optimizer.step()  # params += grad
            self.log({f"train/{k}": losses[k] for k in losses}, on_step=True)
            model.eval()

    def _val_batch(self, batch, batch_idx, loss_calcs):
        params = dict(step=self.state.step, dataset_size=self.state.val_size)
        rs = [calc(batch, *self.models, **params) for calc in loss_calcs]
        return rs

    def val(self, val_loader, loss_calcs):
        if self.is_ddp:
            val_loader = ddp_dataloader(val_loader)

        self.set_state(val_size=len(val_loader.dataset))
        outs = []
        for batch_idx, batch in enumerate(
            tqdm(val_loader, desc=f"Val", disable=self.device != 0)
        ):
            with torch.no_grad():
                batch = self._batch_to_device(batch)
                outs.append(self._val_batch(batch, batch_idx, loss_calcs))
        losses = {k: o[k].mean() for o in default_collate(outs) for k in o}
        if self.is_ddp:
            for k in losses:
                dist.all_reduce(losses[k])
                losses[k] = losses[k] / dist.get_world_size()
        return losses

    def test(self, test_loader, test_batch):
        outs = []
        for batch_idx, batch in enumerate(test_loader):
            with torch.no_grad():
                batch = self._batch_to_device(batch)
                outs.append(test_batch(batch, batch_idx, *self.models))

        collated = collate(outs)
        if self.is_ddp:
            world_size = dist.get_world_size()

            def gather(x):
                tmp = [torch.zeros_like(x) for i in range(world_size)]
                dist.all_gather(tmp, x)
                return torch.vstack(tmp)

            if type(collated) is tuple:
                results = tuple([gather(c) for c in collated])
            elif type(collated) is dict:
                results = {k: gather(collated[k]) for k in collated}
            else:
                results = gather(collated)
        else:
            # torch/utils/data/_utils/collate.py collate_tensor_fn
            results = collated
        return results

    def train(
        self,
        n_epochs,
        loss_calcs,
        train_loader,
        val_loader=None,
        test_loader=None,
        test_batch=None,
        on_test_end=None,
        output_dir=".",
        n_epochs_test_every=1,
        n_epochs_save_every=1,
    ):
        self.set_state(
            train_size=len(train_loader.dataset),
            n_train_batchs=len(train_loader),
        )
        start_epoch = self.state.epoch

        if self.is_ddp:
            train_loader = ddp_dataloader(train_loader)
            val_loader = ddp_dataloader(val_loader)

        for epoch in range(start_epoch, n_epochs):
            self.set_state(epoch=epoch)
            if self.is_ddp:
                train_loader.sampler.set_epoch(epoch)
            for batch_idx, batch in enumerate(
                tqdm(
                    train_loader,
                    desc=f"Epoch {epoch}",
                    disable=self.device != 0,
                )
            ):
                batch = self._batch_to_device(batch)
                self._train_batch(batch, batch_idx, loss_calcs)

            if val_loader is not None:
                losses = self.val(val_loader, loss_calcs)
                self.log({f"val/{k}": losses[k] for k in losses})
            if test_loader is not None and is_time_to_do(
                epoch, n_epochs_test_every
            ):
                test_rs = self.test(test_loader, test_batch)
                if on_test_end is not None:
                    self.log(on_test_end(test_rs, val_loader.dataset, self))

            if is_time_to_do(epoch, n_epochs_save_every):
                self.save(output_dir=output_dir, version=epoch)
            lrs = {
                f"learning_rate_{i}": s.get_last_lr()[0]
                for i, s in enumerate(self.schedulers)
            }
            self.log({**lrs, "epoch": epoch}, commit=True)
