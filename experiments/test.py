from pathlib import Path
import urllib.parse

import numpy as np
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from disentangling.utils.mi import (
    get_mutual_infos,
    get_entropies,
    get_mutual_info,
)
from .utils.ddp import ddp_run
from .utils.create_from_config import (
    create_input_shape,
    create_datasets,
    create_model,
    create_metrics,
)
from .utils.logger import WandbLogger
from .utils.solver import Solver
from .utils.seed_everything import seed_everything


project = "innvariant-representations-eval"


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


def run_metric(metric_fn, y, y_hat, vae, full_samplable_dataset):
    if metric_fn.__name__ == "z_min_var":

        def encode(factors):
            indices = full_samplable_dataset.latent2index(factors)
            dset = Subset(full_samplable_dataset, indices)
            device = next(vae.parameters()).device
            with torch.no_grad():
                rs = [
                    vae.encode(x.to(device)).cpu().numpy()
                    for x, _ in DataLoader(dset, batch_size=256)
                ]
            rs = np.vstack(rs)
            return rs

        sample = lambda n: full_samplable_dataset.sample_latent(n)
        res = metric_fn(
            sample_factors=sample,
            factor2code=encode,
            n_votes=256,
            batch_size=256,
        )
    else:
        if "smoothness" in metric_fn.__name__:
            size = 1024 * 8 * 2
            y, y_hat = y[:size], y_hat[:size]
        discrete_factors = full_samplable_dataset.discrete
        res = metric_fn(y, y_hat, discrete_factors=discrete_factors)
    return res


def run_metrics(metric_funcs, y, y_hat, vae, full_samplable_dataset):
    all_results = {}
    for fn in metric_funcs:
        res = run_metric(fn, y, y_hat, vae, full_samplable_dataset)
        all_results.update(
            {f"{fn.__name__}__{k}": res[k] for k in res}
            if type(res) is dict
            else {fn.__name__: res}
        )
    return all_results


def check_correlation(y, y_hat, discrete_factors=False):
    columns = list(range(y.shape[1]))
    mi_matrix = get_mutual_infos(
        y_hat,
        y,
        estimator="ksg",
        normalized=False,
        discrete_factors=discrete_factors,
    )
    entropies = get_entropies(y, estimator="ksg", discrete=discrete_factors)
    matrices = dict(
        mi_matrix=mi_matrix,
        entropies=entropies.reshape(1, -1),
        mi_matrix_f=get_mutual_infos(
            y_hat,
            y,
            estimator="ksg",
            normalized=False,
            discrete_factors=False,
        ),
    )
    return matrices, columns


def get_on_test_end(metric_funcs):
    def on_test_end(collated, test_set, trainer):
        vae = trainer.models[0]
        y = collated[0].cpu().numpy()
        y_hat = collated[1].cpu().numpy()
        metrics = run_metrics(metric_funcs, y, y_hat, vae, test_set.dataset)
        images = {
            **samples(vae, device=trainer.device),
            **reconstructs(vae, test_set, device=trainer.device),
        }
        # solver.log(*check_correlation(y, y_hat, discrete_factors))
        get_mutual_info.cache.clear()
        return {**images, **metrics}

    return on_test_end


def test_batch(batch, batch_idx, vae, *args, **kwargs):
    x, y = batch
    y_hat = getattr(vae, "module", vae).encode(x)
    return y, y_hat


def init_model(config):
    input_shape = create_input_shape(config.data.name)
    main_model = create_model(
        model_name=config.model_name,
        input_shape=input_shape,
        net_params=config.net_params,
        latent_dim=config.latent_dim,
    )
    return main_model


def run_test(rank, config):
    seed_everything(config.seed)
    group = config.model_name
    name = urllib.parse.urlencode(config.loss_kwargs) or "empty"

    output_dir = (
        Path(config.output_dir)
        / str(config.data.name)
        / str(config.seed)
        / group
        / name
    )

    _, val_set = create_datasets(config.data)
    val_loader = DataLoader(val_set, **config.dataloader.val, pin_memory=True)
    on_test_end = get_on_test_end(metric_funcs=create_metrics(config.metrics))

    logger = WandbLogger(
        is_on_master=rank < 1, n_steps_log_every=config.n_steps_log_every
    )
    logger_params = dict(project=project)

    model = init_model(config.copy())
    solver = Solver([model], ddp_rank=rank, logger=logger)

    wandb_out = output_dir / "eval"
    wandb_out.mkdir(parents=True, exist_ok=True)
    logger.start(
        **logger_params,
        group=group,
        name=name,
        config=config,
        dir=wandb_out,
    )

    collated_dir = output_dir / "collated"
    collated_dir.mkdir(parents=True, exist_ok=True)
    cps_dir = output_dir / "checkpoints"
    for cp_dir in tqdm(
        sorted(cps_dir.iterdir(), key=lambda d: d.stat().st_mtime)
    ):
        solver.resume(cp_dir)
        cache_file = collated_dir / f"{cp_dir.stem}.npz"
        with torch.no_grad():
            collated = solver.test(val_loader, test_batch)
            y = collated[0].cpu().numpy()
            y_hat = collated[1].cpu().numpy()
            np.savez(cache_file, y=y, y_hat=y_hat)
        test_rs = on_test_end(collated, val_loader.dataset, solver)
        solver.log(test_rs, step=solver.state.epoch, commit=True)

    logger.finish()


def test(ddp, config):
    if ddp:
        ddp_run(run_test, config)
    else:
        run_test(-1, config)
