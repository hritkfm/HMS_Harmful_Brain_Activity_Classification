from pathlib import Path
from typing import List, Optional, Tuple, Union, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from scipy.signal import butter, lfilter, sosfiltfilt

from mydatasets.make_eeg_spectrograms import spectrogram_from_eeg


FEATURE_LIST = [
    "Fp1",
    "F3",
    "C3",
    "P3",
    "F7",
    "T3",
    "T5",
    "O1",
    "Fz",
    "Cz",
    "Pz",
    "Fp2",
    "F4",
    "C4",
    "P4",
    "F8",
    "T4",
    "T6",
    "O2",
    "EKG",
]


def butter_lowpass_filter(
    data, cutoff_freq: int = 20, sampling_rate: int = 200, order: int = 4
):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data


def sos_lowpass_filter(data, cutoff_freq, fs=200, order=4):
    # https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter/48677312#48677312
    normal_cutoff = cutoff_freq / (0.5 * fs)
    sos = butter(order, normal_cutoff, btype="low", analog=False, output="sos")
    filtered_data = sosfiltfilt(sos, data)
    return filtered_data


class HMS1DDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        spec_dir_path: str,
        eeg_dir_path: str,
        mode: str = "train",
        fold: int = 0,
        k_fold: int = 4,
        transforms: A.Compose = None,
        sample_rate: int = 200,
        use_kspec: bool = False,
        normalize_method: str = "all",  # "all" か "each"を設定
        vote_min_thresh: int = 0,  # vote数がこの値以下のものは学習に使用しない。(val時は使用)
        without_3votes: bool = False,  # vote数が3のものは使用しない(val時は使用)
        without_miss_seizure: bool = False,  # Seizureのmissラベルと思われるデータを使用しない(valでは使用)
        eeg_nan_ratio_thresh: float = 1.0,  # eegデータのnanの割合がこの値以上のデータは使用しない。(val時は使用)
        spec_nan_ratio_thresh: float = 1.0,  # nanの割合がこの値以上のデータは使用しない。(val時は使用)
        feature_type: str = "standard",  # "half", "standard", "double"
        downsample: int = 1,
        cutoff_freq: int = 20,
        lowpass_filter: str = "lfilter",
        label_type: str = "Classification",  # "Classification": 分類用のラベル, "SED": sed用のラベル
        label_smoothing_ver: str = None,
        label_smoothing_k: int = 10,
        label_smoothing_epsilon: float = 0.05,
        label_smoothing_n_evaluator: int = 3,
        for_pseudo_label: bool = False,  # pseudo_labelを作成するとき用の学習をするか。(fold=0のときはtest:0, val:1, train:2,3)
        tuh_thsz_dir: str = "../../hms-harmful-brain-activity-classification/tuh-thsz",
        extra_data_fold: int | None = None,
        wave_clip_value: int = 1024,
    ):
        self.csv_path = Path(csv_path)
        self.spec_dir_path = Path(spec_dir_path)
        self.eeg_dir_path = Path(eeg_dir_path)
        self.mode = mode
        self.fold = fold
        self.k_fold = k_fold
        self.transforms = transforms
        self.normalize_method = normalize_method
        self.use_kspec = use_kspec
        self.downsample = downsample
        self.sample_rate = sample_rate / downsample
        self.cutoff_freq = cutoff_freq
        self.label_smoothing_ver = label_smoothing_ver
        self.label_smoothing_k = label_smoothing_k
        self.label_smoothing_epsilon = label_smoothing_epsilon
        self.label_smoothing_n_evaluator = label_smoothing_n_evaluator
        self.for_pseudo_label = for_pseudo_label
        # self.spec_tuh_thsz_dir_path = Path(tuh_thsz_dir) / "eeg_10min_spec"
        # self.eeg_tuh_thsz_dir_path = Path(tuh_thsz_dir) / "eeg_raw"
        self.spec_tuh_thsz_dir_path = None
        self.eeg_tuh_thsz_dir_path = None
        self.extra_data_fold = extra_data_fold
        self.lowpass_filter = lowpass_filter
        self.wave_clip_value = wave_clip_value

        ## 使用する信号の種類(kernel準拠)
        # self.use_features = ['Fp1','T3','C3','O1','Fp2','C4','T4','O2']
        # self.feature_to_index = {x:y for x,y in zip(self.use_features, range(len(self.use_features)))}
        if feature_type == "half":
            self.use_features_pair = [
                ["Fp1", "T3"],
                ["T3", "O1"],
                ["Fp1", "C3"],
                ["C3", "O1"],
                ["Fp2", "T4"],
                ["T4", "O2"],
                ["Fp2", "C4"],
                ["C4", "O2"],
            ]
        elif feature_type == "standard":
            self.use_features_pair = [
                ["Fp1", "F7"],
                ["F7", "T3"],
                ["T3", "T5"],
                ["T5", "O1"],
                ["Fp1", "F3"],
                ["F3", "C3"],
                ["C3", "P3"],
                ["P3", "O1"],
                ["Fp2", "F8"],
                ["F8", "T4"],
                ["T4", "T6"],
                ["T6", "O2"],
                ["Fp2", "F4"],
                ["F4", "C4"],
                ["C4", "P4"],
                ["P4", "O2"],
            ]
        elif feature_type == "double":
            self.use_features_pair = [
                ["Fp1", "F7"],
                ["F7", "T3"],
                ["T3", "T5"],
                ["T5", "O1"],
                ["Fp1", "F3"],
                ["F3", "C3"],
                ["C3", "P3"],
                ["P3", "O1"],
                ["Fp1", "Fz"],
                ["Fz", "Cz"],
                ["Cz", "Pz"],
                ["Pz", "O1"],
                ["Fp2", "F8"],
                ["F8", "T4"],
                ["T4", "T6"],
                ["T6", "O2"],
                ["Fp2", "F4"],
                ["F4", "C4"],
                ["C4", "P4"],
                ["P4", "O2"],
                ["Fp2", "Fz"],
                ["Fz", "Cz"],
                ["Cz", "Pz"],
                ["Pz", "O2"],
            ]
        else:
            raise NotImplementedError

        self.df = pd.read_csv(self.csv_path)
        self.has_duplicated_eeg_id = self.df["eeg_id"].duplicated().sum() > 0

        if fold is not None:
            self.df = self.make_fold(self.df, fold)

        # データの除外処理
        if self.mode == "train":
            if without_3votes:
                self.df = self.df[self.df["n_votes"] != 3]

            if without_miss_seizure:
                self.df = self.df[self.df["miss_seizure"] == False]

            self.df = self.df[self.df["n_votes"] >= vote_min_thresh]
            self.df = self.df[self.df["eeg_nan_ratio"] <= eeg_nan_ratio_thresh]
            self.df = self.df[self.df["spec_nan_ratio"] <= spec_nan_ratio_thresh]
            self.df = self.df[self.df["eeg_std_0"] == False].reset_index(drop=True)

        # eeg_idに重複があるかどうか。ある場合はoffsetに従いデータを読み込む
        if self.has_duplicated_eeg_id:  # uniqueなeeg_idで読み込む
            self.unique_eeg_id = self.df.eeg_id.unique()

        self.map_f2n = {f: n for f, n in zip(FEATURE_LIST, range(len(FEATURE_LIST)))}

        # EEGのロード方法の決定(事前にメモリにロードしておくか、ストレージから逐一ロードか)
        eeg_npy_path = self.eeg_dir_path.parent / "train_eegs.npy"
        # if Path(eeg_npy_path).exists():
        if False:  # 遅い・・・
            print("---------------------------")
            print("   USE cache EEG   ")
            print("---------------------------")
            self.eegs = np.load(eeg_npy_path, allow_pickle=True).item()
            self.load_eeg = self._load_eeg_from_memory
        else:
            self.load_eeg = self._load_eeg_from_storage

        # spectrogramのロード方法の決定(事前にメモリにロードしておくか、ストレージから逐一ロードか)
        if use_kspec:
            spec_npy_path = self.spec_dir_path.parent / "train_spec.npy"
            if Path(spec_npy_path).exists():  # train_spec.npyファイルがある場合
                print("---------------------------")
                print("   USE cache spectrogam    ")
                print("---------------------------")
                self.spectrograms = np.load(spec_npy_path, allow_pickle=True).item()
                self.load_spectrogram = self._load_spec_from_memory
            else:
                self.load_spectrogram = self._load_spec_from_storage

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
            self.has_duplicated_eeg_id and self.mode != "test"
        ):  # 重複eeg_idを使いtestでない場合
            return len(self.unique_eeg_id)

        return len(self.df)

    def _load_spec_from_memory(self, spec_id, offset=0):  # offsetは周波数で変換済み
        spec = self.spectrograms[spec_id]
        t_max = spec.shape[-1] - 300
        offset = np.clip(offset, 0, t_max)
        return spec[:, offset : offset + 300]  # (Hz, Time)

    def _load_spec_from_storage(self, spec_id, offset=0):  # offsetは周波数で変換済み
        try:
            spec_id = int(spec_id)
            spec = pd.read_parquet(self.spec_dir_path / f"{spec_id}.parquet")
        except:
            spec = pd.read_parquet(self.spec_tuh_thsz_dir_path / f"{spec_id}.parquet")

        spec = spec.fillna(0).values[:, 1:].T.astype("float32")  # (Hz, Time)
        t_max = spec.shape[-1] - 300
        offset = np.clip(offset, 0, t_max)
        return spec[:, offset : offset + 300]  # (Hz, Time)

    def _eeg_preprocess(self, eeg):
        # === Convert to numpy ===
        data = np.zeros(
            (10_000, len(FEATURE_LIST)), dtype=np.float32
        )  # create placeholder of same shape with zeros
        for index, feature in enumerate(FEATURE_LIST):
            n = self.map_f2n[feature]
            x = eeg[:, n]  # convert to float32
            mean = np.nanmean(
                x
            )  # arithmetic mean along the specified axis, ignoring NaNs
            nan_percentage = np.isnan(x).mean()  # percentage of NaN values in feature
            # === Fill nan values ===
            if nan_percentage < 1:  # if some values are nan, but not all
                x = np.nan_to_num(x, nan=mean)
            else:  # if all values are nan
                x[:] = 0
            data[:, index] = x
        return data

    def _load_eeg_from_memory(self, eeg_id, offset=0):  # offsetは周波数で変換済み
        eeg = self.eegs[f"{eeg_id}"]

        eeg = eeg[offset : offset + 10_000]
        return self._eeg_preprocess(eeg)

    def _load_eeg_from_storage(self, eeg_id, offset=0):  # offsetは周波数で変換済み
        try:
            eeg_id = int(eeg_id)
            eeg = pd.read_parquet(self.eeg_dir_path / f"{eeg_id}.parquet")
        except:  # intに変換できない場合は,tuh_thszデータとして処理
            eeg = pd.read_parquet(self.eeg_tuh_thsz_dir_path / f"{eeg_id}.parquet")
        eeg = eeg.iloc[offset : offset + 10_000]
        # eeg = pd.read_parquet(
        #     self.eeg_dir_path / f"{eeg_id}_{offset // self.EEG_HZ}.parquet"
        # )
        eeg = eeg.values.astype("float32")
        return self._eeg_preprocess(eeg)

    def _get_row(self, index):
        if self.has_duplicated_eeg_id and self.mode != "test":
            if self.mode == "train":
                try:
                    return (
                        self.df[self.df["eeg_id"] == self.unique_eeg_id[index]]
                        .sample(n=1)
                        .iloc[0]
                    )
                except ValueError as e:
                    print("error: eeg_id = ", self.unique_eeg_id[index])
                    print(e)

            else:
                tmp = self.df[
                    self.df["eeg_id"] == self.unique_eeg_id[index]
                ].reset_index(drop=True)
                choise = len(tmp) // 2
                return tmp.iloc[choise]
                # return tmp.iloc[0]

        else:
            return self.df.iloc[index]

    def do_custom_label_smoothing(self, label, epsilon):
        # ピークが複数ある場合、全てのピークを落としてそれ以外を持ち上げる
        max_value = np.max(label)
        idx_max = np.where(label == max_value)[0]
        is_max = np.identity(6)[idx_max].sum(axis=0).astype("bool")
        n_idx_max = is_max.sum()
        new_epsilon = epsilon / n_idx_max
        label[is_max] -= new_epsilon
        label[~is_max] += epsilon / (6 - n_idx_max)
        return label

    def label_smoothing(self, label, n_evaluator):
        if self.label_smoothing_ver == "ver_1":
            epsilon = 1 / (self.label_smoothing_k + np.sqrt(n_evaluator))
            label = self.do_custom_label_smoothing(label, epsilon)
        elif self.label_smoothing_ver == "ver_2":
            if n_evaluator <= self.label_smoothing_n_evaluator:
                label = self.do_custom_label_smoothing(
                    label, self.label_smoothing_epsilon
                )
        return label

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
            if self.has_duplicated_eeg_id:
                spec_offset = row["spectrogram_label_offset_seconds"] * self.SPEC_HZ
            else:
                spec_offset = int(
                    (row["min"] + row["max"]) // 2 * self.SPEC_HZ
                )  # [s] * [hz]
        else:
            if self.has_duplicated_eeg_id:
                spec_offset = (
                    row[
                        "spectrogram_label_offset_seconds"
                    ]  # + np.random.randint(-5, 5)
                ) * self.SPEC_HZ
            else:
                spec_offset = (
                    np.random.randint(row["min"], row["max"] + 1) * self.SPEC_HZ
                )  # [s] * [hz]

        spec = self.load_spectrogram(spec_id, offset=int(spec_offset))  # (Hz, Time)

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

    def _get_eeg(self, row):
        eeg_id = row["eeg_id"]
        offset = 0
        if self.has_duplicated_eeg_id:  # 重複eeg_idを使いtestでない場合
            offset = row["eeg_label_offset_seconds"] * self.EEG_HZ

        X = np.zeros((10_000, len(self.use_features_pair)), dtype="float32")
        data = self.load_eeg(eeg_id, int(offset))
        # === Feature engineering ===
        for i, f_pair in enumerate(self.use_features_pair):
            f1, f2 = f_pair
            X[:, i] = data[:, self.map_f2n[f1]] - data[:, self.map_f2n[f2]]

        # X[:,0] = data[:,self.feature_to_index['Fp1']] - data[:,self.feature_to_index['T3']]
        # X[:,1] = data[:,self.feature_to_index['T3']] - data[:,self.feature_to_index['O1']]

        # X[:,2] = data[:,self.feature_to_index['Fp1']] - data[:,self.feature_to_index['C3']]
        # X[:,3] = data[:,self.feature_to_index['C3']] - data[:,self.feature_to_index['O1']]

        # X[:,4] = data[:,self.feature_to_index['Fp2']] - data[:,self.feature_to_index['C4']]
        # X[:,5] = data[:,self.feature_to_index['C4']] - data[:,self.feature_to_index['O2']]

        # X[:,6] = data[:,self.feature_to_index['Fp2']] - data[:,self.feature_to_index['T4']]
        # X[:,7] = data[:,self.feature_to_index['T4']] - data[:,self.feature_to_index['O2']]

        # === Butter Low-pass Filter ===
        if self.lowpass_filter == "sosfilter":
            X = sos_lowpass_filter(X, cutoff_freq=self.cutoff_freq)
        else:
            X = butter_lowpass_filter(X, cutoff_freq=self.cutoff_freq)

        # === Standarize ===
        if self.wave_clip_value > 0:
            X = np.clip(X, -self.wave_clip_value, self.wave_clip_value)
        X = np.nan_to_num(X, nan=0) / 32.0
        return X[:: self.downsample].astype(np.float32)  # (samples, ch)

    def get_sed_label(self, row, data_type="spectrogram"):
        # 1: 同じ[spectrogram/eeg]_idの行を抽出
        # 2: offsetが[spec: ±300s, eeg: ±25s]以内のものを集める。
        # 3: 長さ[spec: 300, eeg: 50]の配列を作り、各行のラベルをうめていく
        # 4: specの場合中央の256のみを取得(k_specのデータを作成するときと同じ)

        if data_type == "spectrogram":
            offset_range = 600 / 2  # 600秒
            label_length = int(600 * self.SPEC_HZ)
            points_per_sec = 0.5  # 1秒あたりの点数

        else:  # "eeg"
            offset_range = 50 / 2  # 50秒
            label_length = 50  # eeg_specを作成したときのhop_lengthと同じ
            points_per_sec = 50 / 50  # 1秒あたりの点数
        # 1
        id = row[data_type + "_id"]
        rows = self.df[self.df[data_type + "_id"] == id].copy()
        # print(id)

        # 2
        column = f"{data_type}_label_offset_seconds"

        rows.loc[:, column] -= row[column]
        rows = rows[(rows[column] > -offset_range) & (rows[column] <= offset_range)]

        # 3
        sed_label = np.zeros((label_length, len(self.CLASSES)))
        offsets = rows[column].values
        labels = rows[self.CLASSES].values
        mid = int(label_length // 2)
        points_10sec = int(
            10 * points_per_sec
        )  # 1行のラベルで10秒分を埋めるための長さ指定
        # 現状は、ラベルが重複している場合、後ろの行で上書きするようになっている。
        # TODO: label smoothingの処理を入れる。
        for offset, l in zip(offsets, labels):
            offset_points = int(offset * points_per_sec)
            sed_label[mid + offset_points : mid + offset_points + points_10sec] = l

        # 4
        if data_type == "spectrogram":
            sed_label = sed_label[22:-22]
        return sed_label  # (time, n_classes) spec: (256, 6), eeg: (50, 6)

    def __getitem__(self, index: int):
        # train_eeg_aggregate_fold.csvはeeg_idごとにラベルを集約しているので、indexはeeg_idを指定している。

        row = self._get_row(index)
        eeg = self._get_eeg(row)

        # kaggle_spectrograms
        if self.use_kspec:
            Kspec = self._get_Kspec(row)

        if self.transforms:
            eeg = self.transforms(eeg.T, sample_rate=self.sample_rate).T
            # Reverse変換後の配列をTensorに変換するときにnegative slice エラーが出るための対策
            eeg = eeg.copy()

        # label
        if self.mode != "test":
            label = row[self.CLASSES].values.astype("float32")
            eeg_sed_label = self.get_sed_label(row, data_type="eeg")

            if self.mode == "train":
                label = self.label_smoothing(label, row.n_votes)
            output = {"eeg": eeg, "label": label, "eeg_sed_label": eeg_sed_label}
            if self.use_kspec:
                Kspec = np.transpose(Kspec, (2, 0, 1))
                output["Kspec"] = Kspec
            # eeg               : (Time, Ch)
            # target            : (n_classes)
            # eeg_sed_label         : (50, n_classes)
            # (option) Kspec    : (c, freq, time)
            # print(eeg.shape, dtype)
            return output
        else:
            output = {"eeg": eeg}
            if self.use_kspec:
                Kspec = np.transpose(Kspec, (2, 0, 1))
                output["Kspec"] = Kspec
            # eeg               : (Time, Ch)
            # (option) Kspec    : (c, freq, time)
            return output

    def make_fold(self, df, fold):
        folds = np.arange(self.k_fold)
        folds = np.roll(folds, -fold)  # 先頭をfoldにする。

        if self.mode == "train":
            train_fold = folds[1 + int(self.for_pseudo_label) :]
            if self.extra_data_fold is not None:
                train_fold = np.append(train_fold, self.extra_data_fold)
            df_new = df[df.fold.isin(train_fold)].reset_index(drop=True)
            # df_new = df[df.fold != fold].reset_index(drop=True)
        # elif self.mode == "val":
        else:
            val_fold = folds[
                int(self.for_pseudo_label) : int(self.for_pseudo_label) + 1
            ]
            df_new = df[df.fold.isin(val_fold)].reset_index(drop=True)
            # df_new = df[df.fold == fold].reset_index(drop=True)
        # else:
        #     df_new = df
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
