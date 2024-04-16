import shutil
from pathlib import Path

import albumentations
import losses
import models
import numpy as np
import torch
import yaml
import wandb
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import WandbLogger

from utils.color import Color


def train(
    cfg,
    datamodule,
    dataset_args,
    module,
    ckpt_name="{epoch:02}-{val_loss:.3f}",
    module_kwargs={},
):
    assert cfg.logger.mode in ["online", "offline", "debug"]
    gpu = cfg.gpu
    torch.cuda.set_device(gpu)
    seed = np.random.randint(65535) if cfg.seed is None else cfg.seed
    seed_everything(seed)

    if cfg.logger.mode == "debug":
        print(
            Color.RED + "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        )
        print("Debug Mode")
        print(
            "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            + Color.RESET
        )
        # バックプロパゲーションがうまくいくか検査。debug時のみ
        torch.autograd.set_detect_anomaly(True)
        logger = False
        callbacks = []

    else:
        logger = WandbLogger(
            name=cfg.logger.runName,
            version=f"{wandb.util.generate_id()}-" + cfg.logger.runName,
            project=cfg.logger.project,
            offline=True if cfg.logger.mode == "offline" else False,
            log_model=False,
        )
        logger.experiment.config.update(dict(cfg))
        logger.experiment.config.update({"dir": logger.experiment.dir})

        # ckpt_dir = Path(logger.experiment.dir) / "checkpoints"
        # ckpt_dir.mkdir(exist_ok=True, parents=True)
        folder_name = Path(logger.experiment.dir).parent.name
        ckpt_dir = Path("../checkpoints") / f"{folder_name}"
        ckpt_dir.mkdir(exist_ok=True, parents=True)

        checkpoint = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=ckpt_name,
            save_top_k=cfg.checkpoint.save_top_k,
            save_last=cfg.checkpoint.save_last,
            save_weights_only=cfg.checkpoint.save_weights_only,
            every_n_epochs=cfg.checkpoint.every_n_epochs,
            verbose=True,
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
        )
        lr_monitor = LearningRateMonitor(logging_interval=None)
        callbacks = [checkpoint, lr_monitor]

        OmegaConf.save(cfg, ckpt_dir / "train_config.yaml")

    if getattr(cfg.train, "stochastic_weight_averaging", False):
        swa = StochasticWeightAveraging(swa_epoch_start=0.8, swa_lrs=cfg.train.swa_lrs)
        callbacks.append(swa)

    trainer = Trainer(
        logger=logger,
        max_epochs=cfg.train.epoch,
        max_steps=cfg.train.step if "step" in cfg.train.keys() else -1,
        accumulate_grad_batches=cfg.train.n_accumulations,
        limit_val_batches=1.0,
        val_check_interval=cfg.train.val_check_interval,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        devices=[gpu],
        callbacks=callbacks,
        precision="16-mixed" if cfg.train.amp else 32,
        log_every_n_steps=10,
        gradient_clip_val=cfg.train.gradient_clip_val,
        deterministic=cfg.train.deterministic,
        # fast_dev_run=10 if cfg.logger.mode == "debug" else False,
    )
    torch.use_deterministic_algorithms(cfg.train.deterministic)

    if cfg.model.args is None:
        cfg.model.args = {}
    net = getattr(models, cfg.model.name)(**cfg.model.args)

    if cfg.model.load_checkpoint is not None:
        ckpt = torch.load(cfg.model.load_checkpoint, map_location=f"cuda:{gpu}")[
            "state_dict"
        ]
        ckpt = {k[k.find(".") + 1 :]: v for k, v in ckpt.items()}
        missing_keys, unexpected_keys = net.load_state_dict(ckpt, strict=False)
        print(f"\nload checkpoint: {cfg.model.load_checkpoint}\n")
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("====================================")
            print("missing_keys:", missing_keys)
            print("unexpecte_keys:", unexpected_keys)
            print("====================================")

    if cfg.get("loss", None) is not None:
        if "name" in cfg.loss.keys():
            loss_args = (
                dict(cfg.loss.args)
                if "args" in cfg.loss.keys() and cfg.loss.args is not None
                else {}
            )
            for k, v in loss_args.items():
                if k in ["weight", "pos_weight"]:
                    loss_args[k] = torch.tensor(v)
            loss = getattr(losses, cfg.loss.name)(**loss_args)
        else:
            loss = {}
            for loss_key, loss_dict in cfg.loss.items():
                loss_args = (
                    dict(loss_dict.args)
                    if "args" in cfg.loss.keys() and cfg.loss.args is not None
                    else {}
                )
                for k, v in loss_args.items():
                    if k in ["weight", "pos_weight"]:
                        loss_args[k] = torch.tensor(v)
                loss[loss_key] = getattr(losses, loss_dict.name)(**loss_args)
    else:
        loss = None

    data_module = datamodule(
        train_args=dataset_args["train"],
        val_args=dataset_args["val"],
        **cfg.datamodule,
    )

    model = module(
        model=net,
        loss=loss,
        optimizer_name=cfg.optimizer.name,
        optimizer_args=cfg.optimizer.args,
        scheduler=cfg.optimizer.scheduler.name,
        scheduler_args=cfg.optimizer.scheduler.args,
        lr_dict_param=cfg.optimizer.scheduler.lr_dict_param,
        freeze_start=cfg.model.freeze_start.target_epoch,
        unfreeze_params=cfg.model.freeze_start.unfreeze_params,
        **module_kwargs,
    )

    model.datamodule = data_module

    trainer.fit(model, data_module)

    if cfg.logger.mode != "debug":
        logger.finalize('success')
        for n, model_path in enumerate(Path(ckpt_dir).glob("*.ckpt")):
            name = f"model_{n}"
            logger.experiment.config.update({name: model_path.resolve()})
