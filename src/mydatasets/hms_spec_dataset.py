from pathlib import Path
from typing import List, Optional, Tuple, Union, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A

from mydatasets.make_eeg_spectrograms import spectrogram_from_eeg


class HMSHBACSpecDataset(Dataset):
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
        add_mask: bool = False,
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
        self.add_mask = add_mask

        assert all(
            [t in ["kaggle_spec", "eeg_spec"] for t in type_of_data]
        ), "Only 'kaggle_spec' or 'eeg_spec' can be specified for type_of_data."
        self.use_Kspec = True if "kaggle_spec" in type_of_data else False
        self.use_Espec = True if "eeg_spec" in type_of_data else False

        df = pd.read_csv(self.csv_path)
        # eeg_idに重複がある場合は、offsetごとに別のファイルを呼び出す。
        self.load_for_each_offset = df["eeg_id"].duplicated().sum() > 0
        self.df = self.make_fold(df, fold)
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
                # spec_offset = (
                #     row["spectrogram_label_offset_seconds"] + np.random.randint(-5, 5)
                # ) * self.SPEC_HZ
                spec_offset = (row["spectrogram_label_offset_seconds"]) * self.SPEC_HZ
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

    def _get_mask(self):
        kmask = np.zeros((128, 256, 4), dtype=np.float32)
        emask = np.zeros((128, 256, 4), dtype=np.float32)
        kmask[:, 128 - 3 : 128 + 3] = 1.0
        emask[:, 128 - 26 : 128 + 26] = 1.0
        return kmask, emask

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

        Kmask, Emask = self._get_mask()

        if self.use_Kspec and self.use_Espec:
            spec = np.concatenate([Kspec, Espec], axis=-1)  # (128, 256, 8)
            mask = np.concatenate([Kmask, Emask], axis=-1)
        elif self.use_Kspec and not self.use_Espec:
            spec = Kspec  # (128, 256, 4)
            mask = Kmask
        elif not self.use_Kspec and self.use_Espec:
            spec = Espec  # (128, 256, 4)
            mask = Emask
        else:
            raise NotImplementedError

        if self.transforms:
            data = self.transforms(image=spec, mask=mask)
            spec = data["image"]  # (c, hz, time)
            mask = data["mask"]
        else:
            spec = np.transpose(spec, (2, 0, 1))
            mask = np.transpose(mask, (2, 0, 1))

        if self.add_mask:
            spec = np.concatenate([spec, mask], axis=0)  # (c, hz, Time)
        # if not self.channel_stack:  # 一枚画像として入力する場合
        #     c, hz, time = spec.shape
        #     # 4chごとに行方向に積み上げ、その後列方向に連結する。
        #     reshaped_blocks = [
        #         spec[i : i + 4].reshape(1, 4 * hz, time) for i in range(0, c, 4)
        #     ]
        #     spec = np.concatenate(reshaped_blocks, axis=2)

        if self.mode != "test":
            label = row[self.CLASSES].values.astype("float32")
            # data: (ch, Hz, Time)
            return {"data": spec, "target": label}
        else:
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
