import itertools
import json
import wandb

import numpy as np
import optimizers
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F

# from augmentations import cutmix_data, mixup_criterion, mixup_data
from modules.base_module import BaseModule
from omegaconf import OmegaConf
from tqdm import tqdm

from metrics.kaggle_kl_div import score as calc_kl_div
from augmentations.mixup import mixup_data
from augmentations.cutmix import cutmix_data
from augmentations.alphamix import alphamix_data


class HMSHBACSpecModule(BaseModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.validation_step_outputs = []

    def training_step(self, batch, batch_nb):
        image, target = batch["data"], batch["target"]

        if self.transform_train is not None:
            image = self.transform_train(image)

        p = np.random.rand()
        if p < self.p_mix_augmentation:
            method = np.random.choice(self.mix_augmentation)
            if method == "mixup":
                image, target_a, target_b, lam = mixup_data(image, target, alpha=2)
            elif method == "cutmix":
                image, target_a, target_b, lam = cutmix_data(image, target, alpha=0.4)
            elif method == "alphamix":
                image = alphamix_data(image, alpha=2)
                lam = 1
                target_a = target
                target_b = target
            else:
                raise NotImplementedError
            target = lam * target_a + (1 - lam) * target_b

        y = self.forward(image)  # (B, n_classes)
        pred = y["pred"]
        loss = self.loss(pred, target)

        loss_dict = {
            "train_loss": loss,
        }

        self.log_dict(
            loss_dict,
            prog_bar=True,
            logger=True,
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        if self.global_step == 0 and self.logger is not None:
            # 最小のlossを記録
            wandb.define_metric("val_loss", summary="min")

        image, target = batch["data"], batch["target"]

        if self.transform_val is not None:
            image = self.transform_val(image)

        y = self.forward(image)  # (B, n_classes)
        pred = y["pred"]
        loss = self.loss(pred, target)

        loss_dict = {
            "val_loss": loss,
        }

        self.log_dict(
            loss_dict,
        )

        output = {
            "y_pred": F.softmax(pred, dim=1),
            "target": target,
        }

        self.validation_step_outputs.append(output)
        return

    def on_validation_epoch_end(self):
        y_pred = torch.cat(
            [output["y_pred"] for output in self.validation_step_outputs]
        )
        target = torch.cat(
            [output["target"] for output in self.validation_step_outputs]
        )

        # print("y_pred shape:", y_pred.shape)
        # print("target shape:", target.shape)

        y_pred = y_pred.detach().cpu().numpy().astype(np.float32)
        target = target.detach().cpu().numpy().astype(np.float32)

        y_pred_df = pd.DataFrame(y_pred)
        # if not np.allclose(y_pred_df.sum(axis=1), 1):  # エラーが出そうな場合は合計1に再度正規化
        #     y_pred_df = y_pred_df.div(y_pred_df.sum(axis=1), axis=0)

        y_pred_df["id"] = np.arange(len(y_pred_df))

        target_df = pd.DataFrame(target)
        target_df["id"] = np.arange(len(target_df))

        try:
            cv = calc_kl_div(
                solution=target_df, submission=y_pred_df, row_id_column_name="id"
            )
        except:
            print("target", target_df)
            print("pred", y_pred_df)
            print("n nan:", target_df.isna().any().sum())

        log_dict = {
            "val_metric_kldiv": cv,
        }

        self.log_dict(log_dict, prog_bar=True)
        self.validation_step_outputs.clear()

    def on_train_epoch_start(self):
        if self.freeze_start and self.unfreeze_params and self.current_epoch == 0:
            self.freeze()
            print("==Freeze Start==")
            print("Unfreeze params:")
            for n, p in self.model.named_parameters():
                if any(param in n for param in self.unfreeze_params):
                    p.requires_grad = True
                    print(f"  {n}")
            self.model.train()

        if self.current_epoch == self.freeze_start:
            print("==Unfreeze==")
            self.unfreeze()

        if (
            self.current_epoch
            == self.trainer.max_epochs - self.last_n_epoch_refined_augment
        ):
            self._set_refined_augment()

    def _set_refined_augment(self):
        self.transform_train = self.transform_val
        self.p_mix_augmentation = 0
        self.datamodule.train_dataset.b_augment = False
