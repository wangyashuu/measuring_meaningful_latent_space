import torch
import wandb


class WandbLogger:
    def __init__(self, is_on_master, n_steps_log_every=1):
        self.is_on_master = is_on_master
        self.n_steps_log_every = n_steps_log_every

    def log(self, data, columns=None, commit=False, on_step=False, step=None):
        if self.is_on_master and (
            not on_step or (step + 1) % self.n_steps_log_every == 0
        ):
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

    def finish(self):
        if self.is_on_master:
            self.run.finish()

    def start(self, **kwargs):
        if self.is_on_master:
            self.run = wandb.init(**kwargs)

    def summary(self, s = {}):
        if self.is_on_master:
            for k, v in s.items():
                self.run.summary[k] = v