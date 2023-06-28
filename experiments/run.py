import os
from pathlib import Path
import argparse
import yaml
from box import Box
from .train import train
from .test import test


parser = argparse.ArgumentParser(description="Generic runner for VAE models")

parser.add_argument(
    "--config-file",
    metavar="FILE",
    help=(
        " Path to the config file. If provided, the config in the file will be"
        " read. (Optional, default: the default config in"
        " experiment/config.yaml)"
    ),
)

parser.add_argument("--seed", type=int, help="seed")
parser.add_argument(
    "--dataset",
    type=str,
    help=(
        "Dataset name, possible values: `cars3d`, `chairs3d`, `dSprites`,"
        " `shapes3d`, `CelebA` and `MNIST`. (Optional, default: dSprites)"
    ),
)
parser.add_argument(
    "--metrics",
    nargs="*",
    type=str,
    help=(
        " A list of metrics, possible element values: `z_min_var`, `sap`,"
        " `mig`, `mig_sup`, `modularity`, `dci`, `dcimig`, `dcii`,"
        " `smoothness`. (Optional, default: z_min_var sap mig mig_sup"
        " modularity dci dcimig dcii smoothness) "
    ),
)

parser.add_argument(
    "--model-name",
    type=str,
    help=(
        "Model name, possible values: `vae`, `beta_vae`, `beta_tcvae`,"
        " `factor_vae`, `info_vae`, `dip_vae`. (Optional, default: beta_vae)"
    ),
)


## model specific parameters
parser.add_argument(
    "--beta",
    type=float,
    help="Model specific parameter: beta factor for beta_vae",
)
parser.add_argument(
    "--tc-loss-factor",
    type=float,
    help="Model specific parameter: tc_loss_factor for beta_tcvae",
)
parser.add_argument(
    "--d-tc-loss-factor",
    type=float,
    help="Model specific parameter: d_tc_loss_factor for factor_vae",
)
parser.add_argument(
    "--lambd", type=float, help="lModel specific parameter:  ambd for info_vae"
)
parser.add_argument(
    "--dip-type",
    type=str,
    help="Model specific parameter:  dip_type for dip_vae",
)
parser.add_argument(
    "--lambda-od",
    type=float,
    help="Model specific parameter: lambda_od for dip_vae",
)


## parse mode
parser.add_argument("--eval", action="store_true")

args = parser.parse_args()


if __name__ == "__main__":
    ## read config from file
    if args.config_file is not None:
        config_dir = Path(args.config_file)
    else:
        config_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    with open(config_dir / "config.yaml", "r") as config_file:
        base_config = Box(yaml.safe_load(config_file))

    ## accept config from args
    args_config = Box()
    if args.seed is not None:
        args_config.seed = args.seed
    if args.dataset is not None:
        args_config.dataset.name = args.dataset
    if args.metrics is not None:
        args_config.metrics = args.metrics
    if args.model_name is not None:
        with open(config_dir / "config_for_vae.yaml", "r") as vae_config_file:
            temp = Box(yaml.safe_load(vae_config_file))
            vae_config = getattr(
                temp, args.model_name, Box({"loss_kwargs": {}})
            )
        if args.model_name == "beta_vae" and args.beta is not None:
            vae_config.loss_kwargs.beta = args.beta
        if args.model_name == "beta_tcvae" and args.tc_loss_factor:
            vae_config.loss_kwargs.tc_loss_factor = args.tc_loss_factor
        if args.model_name == "factor_vae" and args.d_tc_loss_factor:
            vae_config.loss_kwargs.d_tc_loss_factor = args.d_tc_loss_factor
        if args.model_name == "info_vae" and args.lambd:
            vae_config.loss_kwargs.lambd = args.lambd
        if args.model_name == "dip_vae":
            if args.dip_type:
                vae_config.loss_kwargs.dip_type = args.dip_type
            if args.lambda_od:
                vae_config.loss_kwargs.lambda_od = args.lambda_od
                vae_config.loss_kwargs.lambda_d = (
                    10 * vae_config.loss_kwargs.lambda_od
                    if vae_config.loss_kwargs.dip_type == "i"
                    else vae_config.loss_kwargs.lambda_od
                )
        args_config = vae_config + args_config
        args_config.model_name = args.model_name

    if args.eval:
        test(False, base_config + args_config)
    else:
        train(False, base_config + args_config)
