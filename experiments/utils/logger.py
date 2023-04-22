import torch
import wandb


class WandbLogger:
    def __init__(self, device, n_steps_log_every=1, **kwargs):
        if device == 0:
            self.run = wandb.init(**kwargs)
        else:
            self.run = None
        self.device = device
        self.n_steps_log_every = n_steps_log_every

    def log(self, data, columns=None, commit=False, on_step=False, step=None):
        if self.run is not None and (
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
