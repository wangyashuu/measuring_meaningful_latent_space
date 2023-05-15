from pathlib import Path
import urllib.parse
from tqdm import tqdm
import numpy as np
import json
import yaml
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from disentangling.utils.mi import (
    get_mutual_infos,
    get_entropies,
    get_mutual_info,
)
from disentangling.utils.mi_estimator import estimate_mutual_information

from experiments.utils.ddp import ddp_run, ddp_dataloader, ddp_model
from experiments.utils.create_from_config import (
    create_datasets,
    create_model,
    create_metrics,
)
from experiments.utils.logger import WandbLogger
from experiments.utils.solver import Solver


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


def run_metrics(y, y_hat, metric_funcs, vae, dataset, discrete_factors=False):
    all_results = {}
    for fn in metric_funcs:
        if fn.__name__ == "z_min_var":
            def encode(factors):
                indices = dataset.dataset.latent2index(factors)
                dset = Subset(dataset.dataset, indices)
                device = next(vae.parameters()).device
                x, _ = next(iter(DataLoader(dset, batch_size=256)))
                x = x.to(device)
                with torch.no_grad():
                    r = vae.encode(x).cpu().numpy()
                return r

            res = fn(
                sample_factors=lambda n: dataset.dataset.sample_factors(n),
                factor2code=encode,
                n_votes=288,
                batch_size=256,
            )
        else:
            res = fn(y, y_hat, discrete_factors=discrete_factors)
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


def test_batch(batch, batch_idx, vae, *args, **kwargs):
    x, y = batch
    y_hat = getattr(vae, "module", vae).encode(x)
    return y, y_hat


def init_model(model_name, config):
    model = create_model(
        model_name=model_name,
        input_shape=config.data.input_shape,
        net_params=config.net_params,
        latent_dim=config.latent_dim,
    )
    return model


def read_wandb(p):
    wandb_path = Path(p) / "wandb" / "latest-run" / "files"
    with open(wandb_path / "config.yaml", "r") as f:
        d = yaml.safe_load(f)
    config = {
        k: v["value"]
        for k, v in d.items()
        if k not in ["wandb_version", "_wandb"]
    }
    with open(wandb_path / "wandb-summary.json", "r") as f:
        summary = json.load(f)
    summary = {k: v for k, v in summary.items() if k[0] != "_"}
    return config, summary


def run_test(rank, config):
    _, val_set = create_datasets(config.data)
    val_loader = DataLoader(
        val_set, batch_size=256, num_workers=8, pin_memory=True
    )
    if rank >= 0:
        val_loader = ddp_dataloader(val_loader)

    discrete_factors = val_set.dataset.discrete_factors
    metric_funcs = create_metrics(config.metrics)
    logger = WandbLogger(
        is_on_master=rank < 1, n_steps_log_every=config.n_steps_log_every
    )
    logger_params = dict(project="innvariant-representations-eval")

    def on_test_end(collated, solver):
        vae = solver.models[0]
        if solver._is_ddp():
            vae = vae.module
        y, y_hat = collated
        metrics = run_metrics(
            y, y_hat, metric_funcs, vae, val_set, discrete_factors
        )
        images = {
            **samples(vae, device=solver.device),
            **reconstructs(vae, val_loader.dataset, device=solver.device),
        }
        get_mutual_info.cache.clear()
        # solver.log(*check_correlation(y, y_hat, discrete_factors))
        return {**images, **metrics}

    target_dir = Path("output")
    for seed in range(1, 2):
        seed_dir = target_dir / str(seed)
        # for model_dir in seed_dir.iterdir():
        model_dir = seed_dir / config.model_name
        model_name = model_dir.name

        model = init_model(model_name, config.copy())
        if rank >= 0:
            model = ddp_model(model, rank)
        solver = Solver([model], ddp_rank=rank, logger=logger)
        # for seed_dir in model_dir.iterdir():
        #     seed = seed_dir.name
        #     for params_dir in seed_dir.iterdir():

        # params_name = "beta=8.0"
        # params_dir = model_dir / params_name
        # for params_dir in sorted(model_dir.iterdir()):
        #     params_name = params_dir.name
        params_dir = sorted(model_dir.iterdir())[config.params_idx]
        params_name = params_dir.name
        config, summary = read_wandb(params_dir)
        wandb_out = params_dir / "eval"
        wandb_out.mkdir(parents=True, exist_ok=True)
        logger.start(
            **logger_params,
            group=model_name,
            name=params_name,
            config=config,
            dir=wandb_out,
        )
        logger.summary(summary)
        collated_dir = params_dir / "collated"
        collated_dir.mkdir(parents=True, exist_ok=True)
        cps_dir = params_dir / "checkpoints"
        for cp_dir in tqdm(
            sorted(cps_dir.iterdir(), key=lambda d: d.stat().st_mtime)
        ):
            # cp_dir = sorted(cps_dir.iterdir(), key=lambda d: d.stat().st_mtime)[-1]
            solver.resume(cp_dir)
            cache_file = collated_dir / f"{cp_dir.stem}.npz"
            if cache_file.is_file():
                npz = np.load(cache_file)
                y, y_hat = npz["y"], npz["y_hat"]
            with torch.no_grad():
                collated = solver.test(val_loader, test_batch)
                y = collated[0].cpu().numpy()
                y_hat = collated[1].cpu().numpy()
                np.savez(cache_file, y=y, y_hat=y_hat)
            collated = (y, y_hat)
            test_rs = on_test_end(collated, solver)
            solver.log(test_rs, step=solver.state.epoch, commit=True)
        logger.finish()


def test(config):
    ddp_run(run_test, config)
