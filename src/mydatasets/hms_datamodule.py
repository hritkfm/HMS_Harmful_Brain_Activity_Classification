
import pandas as pd
import pytorch_lightning as pl
import torch

# from mydatasets.rsna_dataset_cls_2d_bone import RSNACls2DBoneDataset
from mydatasets.hms_spec_dataset import HMSHBACSpecDataset
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler


class HMSHBACSpecDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_args: dict,
        val_args: dict,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        use_sampler: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_args = train_args
        self.val_args = val_args
        self.train_sampler = None
        self.use_sampler = use_sampler

    def set_sampler(self):
        print("-------------------")
        print("    Use sampler    ")
        print("-------------------")
        
        # eeg_idごとのexpert_consensusの最頻値を取得
        expert_consensus_df = self.train_dataset.df.groupby("eeg_id")["expert_consensus"].agg(lambda x: x.mode()[0])
        expert_consensus_df = pd.DataFrame(expert_consensus_df)
        # expert_consensusの1/出現割合を計算
        weight = 1 / (expert_consensus_df.expert_consensus.value_counts() / len(expert_consensus_df))
        weight = dict(weight)
        expert_consensus_df["weight"] = expert_consensus_df["expert_consensus"].map(weight)
        # train_datasetのunique_eeg_idと順番をあわせる
        order_eeg_id = self.train_dataset.unique_eeg_id
        expert_consensus_df = expert_consensus_df.reindex(index=order_eeg_id)
        sample_weight = expert_consensus_df["weight"].values
        # replacement=Trueで少数ラベルデータを重複して取得可能になる。(アップサンプル)
        # replacement=Falseで重複を禁止する。(ダウンサンプル)
        self.train_sampler = WeightedRandomSampler(
            weights=sample_weight, num_samples=len(self.train_dataset), replacement=True
        )

    def setup(self, stage=None):
        self.train_dataset = HMSHBACSpecDataset(**self.train_args)
        self.val_dataset = HMSHBACSpecDataset(**self.val_args)
        if self.use_sampler:
            self.set_sampler()

    def train_dataloader(self):
        loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True if self.train_sampler is None else False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return loader
