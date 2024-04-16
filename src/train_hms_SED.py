import argparse
import copy
from pathlib import Path

import wandb
import models
import numpy as np
import pandas as pd
import torch
from augmentations.augmentation import hms_spec_augmentations
from modules import HMSSEDModule
from mydatasets import HMSSEDDataModule
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from base_train import train


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
        default="./configs/hms_SED.yml",
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

    assert cfg["dataset"]["fold"] is not None

    datamodule = HMSSEDDataModule

    transforms = hms_spec_augmentations(
        cfg.height, cfg.width, augment_args=cfg.transforms
    )
    dataset_args = {
        "train": {
            "mode": "train",
            "transforms": transforms["albu_train"],
            # "transforms": None,
            **cfg.dataset,
        },
        "val": {
            "mode": "val",
            "transforms": transforms["albu_val"],
            # "transforms": None,
            **cfg.dataset,
        },
    }

    module = HMSSEDModule
    module_kwargs = dict(cfg.module) if cfg.module is not None else {}
    module_kwargs.update(
        {
            "transform_train": transforms["torch_train"],
            "transform_val": transforms["torch_val"],
        }
    )

    args = copy.deepcopy(cfg.model)
    args = OmegaConf.to_container(args)
    del args["name"]
    del args["load_checkpoint"]
    del args["freeze_start"]

    ckpt_name = f"{{epoch:02}}-{{{cfg.checkpoint.monitor}:.3f}}"
    train(cfg, datamodule, dataset_args, module, ckpt_name, module_kwargs)



if __name__ == "__main__":
    main()
