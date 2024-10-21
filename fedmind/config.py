import argparse

import yaml

from fedmind.utils import EasyDict


def dotodict(k: str, value):
    """
    convert dot connected keys to recursive dict

    Args:
        k: string of keys like "k1.k2.k3"
        v: final value

    Return:
        dict: {'k1': {'k2': {'k3': v}}}
    """
    keys = k.split(".")[::-1]
    for key in keys:
        value = {key: value}
    return value


def parse_args(ipython: bool = False) -> argparse.Namespace:
    """Parse command line arguments

    Args:
        ipython: whether to run in ipython

    Returns:
        argparse.Namespace: parsed arguments
    """
    parser = argparse.ArgumentParser(description="options for FedLearn")

    # specify custom config file
    parser.add_argument("--cfg", help="custom config file", type=str)

    # specify id for multiple runs with same config
    parser.add_argument("--id", help="id of multiple runs with same config", type=int)

    # modify config options using the command-line in [k1, v1, k2, v2, ...]
    parser.add_argument("opts", help="modify config options", nargs=argparse.REMAINDER)

    return parser.parse_args() if not ipython else parser.parse_args([])


def get_config(conf_path: str, ipython: bool = False) -> EasyDict:
    """Get config from default config file, custom config file and command line

    Args:
        conf_path: path to the default config file
        ipython: whether to run in ipython

    Returns:
        EasyDict: config
    """

    args = parse_args(ipython)

    # 1. load default config file
    with open(conf_path, "r") as f:
        default_config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(default_config)

    # 2. load custom config file
    if args.cfg is not None:
        with open(args.cfg, "r") as f:
            custom_config = yaml.load(f, Loader=yaml.FullLoader)
        # merge default config and custom config
        custom_config = EasyDict(custom_config)
        config.merge(custom_config)

    if args.id is not None:
        config["id"] = args.id

    # 3. modify config options using the command-line
    for i in range(0, len(args.opts), 2):
        config.merge(dotodict(args.opts[i], args.opts[i + 1]))

    return config
