import os
import yaml
import argparse
from box import Box
from .train import train


parser = argparse.ArgumentParser(description="Generic runner for VAE models")
parser.add_argument(
    "--config",
    "-c",
    dest="filename",
    metavar="FILE",
    help="path to the config file",
)
parser.add_argument(
    "--devices",
    "-d",
    dest="devices",
    nargs="+",
    type=int,
    help="gpu devices",
)


args = parser.parse_args()

if __name__ == "__main__":
    config_file = open(args.filename, "r")
    default_config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "config.yaml"
    )
    default_config_file = open(default_config_path, "r")
    config, default_config = [
        Box(yaml.safe_load(f)) for f in [config_file, default_config_file]
    ]
    default_config_file.close()
    config_file.close()
    config = default_config + config
    model = train(config)
