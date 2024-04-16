from pathlib import Path
from typing import List, Optional, Tuple, Union, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A

from mydatasets.make_eeg_spectrograms import spectrogram_from_eeg


class HMSSEDDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        spec_dir_path: str,
        eeg_dir_path: str,
        type_of_data: list[str] = ["kaggle_spec", "eeg_spec"],
        eeg_spec_dir_path: str = None,
        mode: str = "train",
        fold: int = 0,
        k_fold: int = 4,
        transforms: A.Compose = None,
        normalize_method: str = "all",  # "all" か "each"を設定
        vote_min_thresh: int = 0, # vote数がこの値以下のものは学習に使用しない。(val時は使用)
        eeg_nan_ratio_thresh: float = 1.0, # eegデータのnanの割合がこの値以上のデータは使用しない。(val時は使用)
        spec_nan_ratio_thresh: float = 1.0, # nanの割合がこの値以上のデータは使用しない。(val時は使用)
    ):
        self.csv_path = Path(csv_path)
        self.spec_dir_path = Path(spec_dir_path)
        self.eeg_dir_path = Path(eeg_dir_path)
        self.mode = mode
        self.fold = fold
        self.k_fold = k_fold
        self.transforms = transforms
        self.normalize_method = normalize_method
        self.eeg_spec_dir_path = eeg_spec_dir_path

        assert all(
            [t in ["kaggle_spec", "eeg_spec"] for t in type_of_data]
        ), "Only 'kaggle_spec' or 'eeg_spec' can be specified for type_of_data."
        self.use_Kspec = True if "kaggle_spec" in type_of_data else False
        self.use_Espec = True if "eeg_spec" in type_of_data else False

        df = pd.read_csv(self.csv_path)
        # フラグ：eeg_idに重複がある場合は、offsetごとに別のファイルを呼び出す。
        self.load_for_each_offset = df["eeg_id"].duplicated().sum() > 0
        self.df = self.make_fold(df, fold)
        
        # データの除外処理
        if self.mode == "train":
            self.df = self.df[self.df["n_votes"] >= vote_min_thresh]
            self.df = self.df[self.df["eeg_nan_ratio"] <= eeg_nan_ratio_thresh]
            self.df = self.df[self.df["spec_nan_ratio"] <= spec_nan_ratio_thresh]
            self.df = self.df[self.df["eeg_std_0"] == False].reset_index(drop=True)

        if self.load_for_each_offset:  # uniqueなeeg_idで読み込む
            self.unique_eeg_id = self.df.eeg_id.unique()

        # spectrogramのロード方法の決定(事前にメモリにロードしておくか、ストレージから逐一ロードか)
        spec_npy_path = self.spec_dir_path.parent / "train_spec.npy"
        if Path(spec_npy_path).exists():  # train_spec.npyファイルがある場合
            print("---------------------------")
            print("   USE cache spectrogam    ")
            print("---------------------------")
            self.spectrograms = np.load(spec_npy_path, allow_pickle=True).item()
            self.get_spectrogram = self._get_spec_from_memory
        else:
            self.get_spectrogram = self._get_spec_from_storage

        # eeg_specのロード方法を決定(事前にメモリにロードしておくか、ストレージから逐一ロードか)
        if (
            eeg_spec_dir_path is not None
            and (Path(eeg_spec_dir_path) / "all.npy").exists()
        ):
            print("--------------------------------")
            print("    USE cache eeg spectrogam    ")
            print("--------------------------------")
            self.eeg_spectrograms = np.load(
                Path(eeg_spec_dir_path) / "all.npy", allow_pickle=True
            ).item()
            self.get_eeg_spectrogram = self._get_eeg_spec_from_memory
        else:
            self.get_eeg_spectrogram = self._get_eeg_spec_from_storage

        self.CLASSES = [
            "seizure_vote",
            "lpd_vote",
            "gpd_vote",
            "lrda_vote",
            "grda_vote",
            "other_vote",
        ]

        self.SPEC_HZ = 0.5
        self.EEG_HZ = 200

    def __len__(self):
        if (
            self.load_for_each_offset and self.mode != "test"
        ):  # 重複eeg_idを使いtestでない場合
            return len(self.unique_eeg_id)

        return len(self.df)

    def _get_spec_from_memory(self, spec_id, offset=0):
        spec = self.spectrograms[spec_id]
        t_max = spec.shape[-1] - 300
        offset = np.clip(offset, 0, t_max)
        return spec[:, offset : offset + 300]  # (Hz, Time)

    def _get_spec_from_storage(self, spec_id, offset=0):
        spec = pd.read_parquet(self.spec_dir_path / f"{spec_id}.parquet")
        spec = spec.fillna(0).values[:, 1:].T.astype("float32")  # (Hz, Time)
        t_max = spec.shape[-1] - 300
        offset = np.clip(offset, 0, t_max)
        return spec[:, offset : offset + 300]  # (Hz, Time)

    def _get_eeg_spec_from_memory(self, eeg_id):  # 現状、EEG_specのV1専用
        return self.eeg_spectrograms[eeg_id]

    def _get_eeg_spec_from_storage(self, eeg_id, offset_sec=0):
        if self.load_for_each_offset:  # EEG_specのV2以降はこちらを使用
            eeg_spec = np.load(
                self.eeg_spec_dir_path + "/" + f"{eeg_id}_{offset_sec}.npy"
            )
        else:
            eeg_spec = np.load(self.eeg_spec_dir_path + "/" + f"{eeg_id}.npy")
        return eeg_spec

    def _get_row(self, index):
        if self.load_for_each_offset and self.mode != "test":
            if self.mode == "train":
                return (
                    self.df[self.df["eeg_id"] == self.unique_eeg_id[index]]
                    .sample(n=1)
                    .iloc[0]
                )
            else:
                tmp = self.df[
                    self.df["eeg_id"] == self.unique_eeg_id[index]
                ].reset_index(drop=True)
                choise = len(tmp) // 2
                return tmp.iloc[choise]
                # return tmp.iloc[0]

        else:
            return self.df.iloc[index]
        
    def get_sed_label(self, row, label_type="spectrogram"):
        # 1: eeg_idと同じspectrogram_idの行を抽出
        # 2: 各行でspectrogram_label_offsetがプラマイ150s以内のものを集める。
        # 3: 600s*0.5Hz=300のラベルを作成し、各行のラベルでoffsetに従い埋める。(各行のラベルは10s*0.5Hz=5点を埋める)
        # 4: 中央の256のみを取得(k_specのデータを作成するときと同じ)
        if label_type == "spectrogram":
            offset_range = 600 / 2 # 600秒
            label_length = int(600 * self.SPEC_HZ)
            points_per_sec = 0.5 # 1秒あたりの点数

        else: # "eeg"
            offset_range = 50 / 2 # 50秒
            label_length = 256 # eeg_specを作成したときのhop_lengthと同じ
            points_per_sec = 256 / 50 # 1秒あたりの点数
        #1 
        id = row[label_type+"_id"]
        rows = self.df[self.df[label_type+"_id"] == id].copy()
        # print(id)

        #2
        column = f"{label_type}_label_offset_seconds"

        rows.loc[:,column] -= row[column]
        rows = rows[(rows[column] > -offset_range) & (rows[column] <= offset_range)]

        #3
        sed_label = np.zeros((label_length, len(self.CLASSES)))
        offsets = rows[column].values
        labels = rows[self.CLASSES].values
        mid = int(label_length // 2)
        points_10sec = int(10 * points_per_sec)
        for offset, l in zip(offsets, labels):
            offset_points = int(offset * points_per_sec)
            sed_label[mid + offset_points: mid + offset_points + points_10sec] = l 
        
        #4
        if label_type == "spectrogram":
            sed_label = sed_label[22:-22]
        return sed_label # (256, n_classes)
    
    def normalize(self, spec: np.ndarray):
        # log transform
        spec -= spec.min() + np.exp(-4)
        spec = np.clip(spec, np.exp(-4), np.exp(8))
        # spec = np.clip(spec, np.exp(-4), np.exp(12))
        spec = np.log(spec)

        # normalize per image
        eps = 1e-6
        spec_mean = spec.mean(axis=(0, 1))
        spec = spec - spec_mean
        spec_std = spec.std(axis=(0, 1))
        spec = spec / (spec_std + eps)
        return spec

    def _get_Kspec(self, row):
        spec_id = row["spectrogram_id"]
        # offsetに従い、特定の領域を切り出してくる。
        if self.mode == "test":
            spec_offset = 0
        elif self.mode == "val":
            if self.load_for_each_offset:
                spec_offset = row["spectrogram_label_offset_seconds"] * self.SPEC_HZ
            else:
                spec_offset = int(
                    (row["min"] + row["max"]) // 2 * self.SPEC_HZ
                )  # [s] * [hz]
        else:
            if self.load_for_each_offset:
                spec_offset = (
                    row["spectrogram_label_offset_seconds"]# + np.random.randint(-5, 5)
                ) * self.SPEC_HZ
            else:
                spec_offset = (
                    np.random.randint(row["min"], row["max"] + 1) * self.SPEC_HZ
                )  # [s] * [hz]

        spec = self.get_spectrogram(spec_id, offset=int(spec_offset))  # (Hz, Time)

        # log transform
        if self.normalize_method == "all":
            spec = self.normalize(spec)

        else:  # spectrogramは4*100で別々のデータなので、別々に正規化する。
            normalized_spec = [
                self.normalize(spec[st : st + 100]) for st in range(0, 400, 100)
            ]
            spec = np.concatenate(normalized_spec, axis=0)

        # 後の処理のため一度チャネル方向にスタックしておく
        spec = np.stack(
            [spec[st : st + 100] for st in range(0, 400, 100)], axis=-1
        )  # shape: (Hz, Time) -> (Hz, Time, Channel):(100, 300, 4)
        # shape: (100, 300, 4) => (128, 256, 4)
        new_spec = np.zeros((128, 256, 4), dtype="float32")
        new_spec[14:-14] = spec[:, 22:-22]
        return new_spec

    def _get_Espec(self, row):
        if self.eeg_spec_dir_path is not None:
            eeg_offset = 0
            if self.load_for_each_offset:
                eeg_offset = row["eeg_label_offset_seconds"]
            eeg_spec = self.get_eeg_spectrogram(
                row["eeg_id"],
                offset_sec=int(eeg_offset),
            )
        else:  # 事前に保存していない場合は一から作る。以上に遅い上、使わないのでエラーになるようにした
            raise NotImplementedError
            # eeg_spec = spectrogram_from_eeg(
            #     str(self.eeg_dir_path / f"{row['eeg_id']}.parquet"),
            #     denoise_wavelet=None,
            # )
        return eeg_spec  # (hz, time, ch):(128, 256, 4)

    def __getitem__(self, index: int):
        # train_eeg_aggregate_fold.csvはeeg_idごとにラベルを集約しているので、indexはeeg_idを指定している。
        #
        # row = self.df.iloc[index]
        row = self._get_row(index)

        # kaggle_spectrograms
        if self.use_Kspec:
            Kspec = self._get_Kspec(row)

        # EEG_spectrograms
        if self.use_Espec:
            Espec = self._get_Espec(row)

        if self.use_Kspec and self.use_Espec:
            spec = np.concatenate([Kspec, Espec], axis=-1)  # (128, 256, 8)
        elif self.use_Kspec and not self.use_Espec:
            spec = Kspec  # (128, 256, 4)
        elif not self.use_Kspec and self.use_Espec:
            spec = Espec  # (128, 256, 4)
        else:
            raise NotImplementedError


        if self.mode != "test":
            kspec_SED_label = self.get_sed_label(row, label_type="spectrogram")
            if self.transforms:
                data = self.transforms(image=spec, mask=kspec_SED_label.transpose(1, 0))
                spec = data["image"]  # (c, hz, time)
                kspec_SED_label = data["mask"].transpose(1, 0)
            else:
                spec = np.transpose(spec, (2, 0, 1))

            label = row[self.CLASSES].values.astype("float32")
            # data: (ch, Hz, Time), target: (n_classes), target_SED: (256, n_classes)
            return {"data": spec, "target": label, "target_SED": kspec_SED_label}
        else:
            if self.transforms:
                data = self.transforms(image=spec)
                spec = data["image"]  # (c, hz, time)
            else:
                spec = np.transpose(spec, (2, 0, 1))
            return {"data": spec}

    def make_fold(self, df, fold):
        if self.mode == "train":
            df_new = df[df.fold != fold].reset_index(drop=True)
        elif self.mode == "val":
            df_new = df[df.fold == fold].reset_index(drop=True)
        else:
            df_new = df
        return df_new


if __name__ == "__main__":
    csv_path: str = (
        "../../hms-harmful-brain-activity-classification/train_eeg_aggregate_fold.csv"
    )
    spec_dir_path: str = (
        "../../hms-harmful-brain-activity-classification/train_spectrograms"
    )
    spec_npy_path: str = (
        "../../hms-harmful-brain-activity-classification/train_spec.npy"
    )
    spec_npy_path: str = None
    mode: str = "train"
    fold: int = 0
    k_fold: int = 4
    transforms: A.Compose = None

    dataset = HMSHBACSpecDataset(
        csv_path,
        spec_dir_path,
        spec_npy_path,
        mode=mode,
        fold=fold,
        k_fold=k_fold,
        transforms=transforms,
    )

    print("dataset length", len(dataset))

    n = np.random.randint(len(dataset))

    import time

    start = time.time()
    data = dataset[n]
    print(data["data"].shape, data["target"])

    assert data["data"].shape == (400, 300, 1)
    assert len(data["target"]) == 6

    print("elapsed time:", time.time() - start)
