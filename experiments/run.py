import os
from pathlib import Path
import argparse
import yaml
from box import Box
from .train import train

parser = argparse.ArgumentParser(description="Generic runner for VAE models")
parser.add_argument(
    "--config-file",
    metavar="FILE",
    help="path to the config file",
)

parser.add_argument(
    "--model-name",
    type=str,
    help=(
        "model name (vae, beta_vae, beta_tcvae, factor_vae, info_vae, dip_vae)"
    ),
)

parser.add_argument("--beta", type=float, help="beta factor for beta_vae")
parser.add_argument(
    "--tc-loss-factor", type=float, help="tc_loss_factor for beta_tcvae"
)
parser.add_argument(
    "--d-tc-loss-factor", type=float, help="d_tc_loss_factor for factor_vae"
)
parser.add_argument("--lambd", type=float, help="lambd for info_vae")
parser.add_argument("--dip-type", type=float, help="dip_type for dip_vae")
parser.add_argument("--lambda-od", type=float, help="lambda_od for dip_vae")

args = parser.parse_args()

if __name__ == "__main__":
    if args.config_file is not None:
        with open(args.config_file, "r") as config_file:
            config = Box(yaml.safe_load(config_file))
    elif args.model_name is not None:
        config_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        with open(config_dir / "config.yaml", "r") as default_config_file:
            default_config = Box(yaml.safe_load(default_config_file))
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
                    10 * vae_config.lambda_od
                    if vae_config.dip_type == "i"
                    else vae_config.lambda_od
                )
        config = default_config + vae_config
        config.model_name = args.model_name
    model = train(False, config)
