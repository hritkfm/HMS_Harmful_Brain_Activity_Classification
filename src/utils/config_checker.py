# configファイルが適切に読み込めているかチェックする。
import argparse
import copy
from pathlib import Path
from omegaconf import OmegaConf


def allkeys(x):
    for key, value in x.items():
        yield key
        if isinstance(value, dict):
            for child in allkeys(value):
                yield key + "." + child


def check_dotlist(cfg, dotlist):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_keys = list(allkeys(cfg_dict))
    dotlist_dict = OmegaConf.to_container(dotlist, resolve=True)
    dotlist_keys = list(allkeys(dotlist_dict))

    for d_key in dotlist_keys:
        assert d_key in cfg_keys, f"{d_key} dosen't exist in config file."


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        default="./configs/hms_spec.yml",
        help="path of the config file",
    )
    parser.add_argument("--options", "-o", nargs="*", help="optional arguments")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_file)
    if args.options is not None:
        dotlist = OmegaConf.from_dotlist(args.options)
        check_dotlist(cfg, dotlist)
        cfg = OmegaConf.merge(cfg, dotlist)

    if cfg.dataset.fold is not None:
        cfg.logger.runName = f"{cfg.logger.runName}_fold_{cfg.dataset.fold}"
    else:
        cfg.logger.runName = f"{cfg.logger.runName}_all_data"

    return cfg


def main():
    cfg = parser()


if __name__ == "__main__":
    main()
