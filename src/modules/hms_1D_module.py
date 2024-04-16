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
from augmentations.mixup import mixup_data, zebramix_data
from augmentations.cutmix import cutmix_data_v2
from augmentations.alphamix import alphamix_data
from utils.general_utils import log_best


def get_middle_row(group):
    middle_index = len(group) // 2
    return group.iloc[[middle_index]]


class HMS1DModule(BaseModule):
    def __init__(
        self,
        p_reverse: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.validation_step_outputs = []
        self.kldiv_min = log_best(goal="min")
        self.kldiv_low_votes_min = np.inf
        self.kldiv_high_votes_min = np.inf
        self.kldiv_middle_votes_min = np.inf
        self.kldiv_high_votes_Hmin = log_best(goal="min")
        self.kldiv_Hmin = np.inf
        self.kldiv_low_votes_Hmin = np.inf
        self.kldiv_middle_votes_Hmin = np.inf
        self.p_reverse = p_reverse

    def _shared_step(self, batch, transform=None):
        eeg = batch["eeg"]
        kspec = batch.get("Kspec")

        if (transform is not None) and (kspec is not None):
            kspec = transform(kspec)

        y = self.forward(eeg, batch["label"], batch["eeg_sed_label"])
        return y

    def training_step(self, batch, batch_nb):

        p = np.random.rand()
        if p < self.p_mix_augmentation:
            method = np.random.choice(self.mix_augmentation)
            if method == "mixup":
                eeg, y_a, y_b, lam = mixup_data(batch["eeg"], batch["label"], alpha=2)
            elif method == "zebramix":
                eeg, y_a, y_b, lam = zebramix_data(batch["eeg"], batch["label"])
            else:
                raise NotImplementedError
            label = lam * y_a + (1 - lam) * y_b
            batch["eeg"] = eeg
            batch["label"] = label

        # SED追加に伴い、Reverse処理をここで記述(面倒なのでバッチごと反転)
        p = np.random.rand()
        if p < self.p_reverse:
            batch["eeg"] = torch.flip(batch["eeg"], dims=[1])
            batch["eeg_sed_label"] = torch.flip(batch["eeg_sed_label"], dims=[1])

        y = self._shared_step(batch, self.transform_train)
        pred = y["pred"]
        pred_sed = y["pred_sed"]
        target = y["target"]
        # target = batch["label"]

        target_sed = y["target_sed"]
        # target_sed = batch["eeg_sed_label"]
        loss = self.loss(pred, pred_sed, target, target_sed)

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

        y = self._shared_step(batch, self.transform_val)
        pred = y["pred"]
        pred_sed = y["pred_sed"]
        target = batch["label"]

        target_sed = batch["eeg_sed_label"]
        loss = self.loss(pred, pred_sed, target, target_sed)

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
        # if self.global_step == 0 and self.logger is not None:
        #     # 最小値を記録
        #     wandb.define_metric("val_metric_kldiv", summary="min")
        #     wandb.define_metric("val_metric_kldiv_low_votes", summary="min")
        #     wandb.define_metric("val_metric_kldiv_high_votes", summary="min")

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

        # n_votesを取得し、n_votesの多いデータと少ないデータでメトリック計算
        df = self.datamodule.val_dataset.df
        unique_eeg_id = self.datamodule.val_dataset.unique_eeg_id
        # val時はeeg_idごとに中間のデータを使っているため中間のみを取得
        middle_df = df.groupby("eeg_id").apply(get_middle_row).reset_index(drop=True)
        middle_df = middle_df.set_index("eeg_id")
        middle_df = middle_df.reindex(unique_eeg_id)
        n_votes = middle_df["n_votes"].values
        y_pred_df["n_votes"] = n_votes[: len(y_pred)]
        target_df["n_votes"] = n_votes[: len(y_pred)]
        try:
            cv = calc_kl_div(
                solution=target_df.drop("n_votes", axis=1),
                submission=y_pred_df.drop("n_votes", axis=1),
                row_id_column_name="id",
            )
            high_target = target_df[target_df["n_votes"] >= 10].reset_index(drop=True)
            high_pred = y_pred_df[y_pred_df["n_votes"] >= 10].reset_index(drop=True)
            low_target = target_df[target_df["n_votes"] < 10].reset_index(drop=True)
            low_pred = y_pred_df[y_pred_df["n_votes"] < 10].reset_index(drop=True)
            middle_target = target_df[
                (target_df["n_votes"] >= 3) & (target_df["n_votes"] < 20)
            ].reset_index(drop=True)
            middle_pred = y_pred_df[
                (y_pred_df["n_votes"] >= 3) & (y_pred_df["n_votes"] < 20)
            ].reset_index(drop=True)
            cv_low_votes = calc_kl_div(
                solution=low_target.drop("n_votes", axis=1),
                submission=low_pred.drop("n_votes", axis=1),
                row_id_column_name="id",
            )
            cv_high_votes = calc_kl_div(
                solution=high_target.drop("n_votes", axis=1),
                submission=high_pred.drop("n_votes", axis=1),
                row_id_column_name="id",
            )
            cv_middle_votes = calc_kl_div(
                solution=middle_target.drop("n_votes", axis=1),
                submission=middle_pred.drop("n_votes", axis=1),
                row_id_column_name="id",
            )
        except:
            print("target", target_df)
            print("pred", y_pred_df)
            print("n nan:", target_df.isna().any().sum())
        # import pdb; pdb.set_trace()

        self.validation_step_outputs.clear()
        if self.current_epoch >= 0:
            if self.kldiv_min.update(cv):
                self.kldiv_low_votes_min = cv_low_votes
                self.kldiv_high_votes_min = cv_high_votes
                self.kldiv_middle_votes_min = cv_middle_votes
            if self.kldiv_high_votes_Hmin.update(cv_high_votes):
                self.kldiv_Hmin = cv
                self.kldiv_low_votes_Hmin = cv_low_votes
                self.kldiv_middle_votes_Hmin = cv_middle_votes

            log_dict = {
                "val_metric_kldiv": cv,
                "val_metric_kldiv_low_votes": cv_low_votes,
                "val_metric_kldiv_high_votes": cv_high_votes,
                "val_metric_kldiv_middle_votes": cv_middle_votes,
                "val_metric_kldiv_min": self.kldiv_min.best_score,
                "val_metric_kldiv_low_votes_min": self.kldiv_low_votes_min,
                "val_metric_kldiv_high_votes_min": self.kldiv_high_votes_min,
                "val_metric_kldiv_middle_votes_min": self.kldiv_middle_votes_min,
                "val_metric_kldiv_high_votes_Hmin": self.kldiv_high_votes_Hmin.best_score,
                "val_metric_kldiv_Hmin": self.kldiv_Hmin,
                "val_metric_kldiv_low_votes_Hmin": self.kldiv_low_votes_Hmin,
                "val_metric_kldiv_middle_votes_Hmin": self.kldiv_middle_votes_Hmin,
            }

            self.log_dict(log_dict, prog_bar=True)

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
