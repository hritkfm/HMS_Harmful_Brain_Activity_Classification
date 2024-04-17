import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import KFold, GroupKFold


def group_by_eeg_id(train_csv):
    df = pd.read_csv(train_csv)
    TARGETS = df.columns[-6:]
    print("Train shape:", df.shape)
    print("Targets", list(TARGETS))
    train = df.groupby("eeg_id")[
        ["spectrogram_id", "spectrogram_label_offset_seconds"]
    ].agg({"spectrogram_id": "first", "spectrogram_label_offset_seconds": "min"})
    train.columns = ["spectrogram_id", "min"]

    tmp = df.groupby("eeg_id")[
        ["spectrogram_id", "spectrogram_label_offset_seconds"]
    ].agg({"spectrogram_label_offset_seconds": "max"})
    train["max"] = tmp

    tmp = df.groupby("eeg_id")[["patient_id"]].agg("first")
    train["patient_id"] = tmp

    tmp = df.groupby("eeg_id")[TARGETS].agg("sum")
    for t in TARGETS:
        train[t] = tmp[t].values

    y_data = train[TARGETS].values
    y_data = y_data / y_data.sum(axis=1, keepdims=True)
    train[TARGETS] = y_data

    tmp = df.groupby("eeg_id")[["expert_consensus"]].agg("first")
    train["target"] = tmp

    train = train.reset_index()
    print("Train non-overlapp eeg_id shape:", train.shape)
    return train


def get_foldv1(train_v1):
    all_oof = []
    all_true = []
    TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}

    train_v1["fold"] = 0
    gkf = GroupKFold(n_splits=4)
    for i, (train_index, valid_index) in enumerate(
        gkf.split(train_v1, train_v1.target, train_v1.patient_id)
    ):
        train_v1.loc[valid_index, "fold"] = i
    return train_v1


def get_foldv2(train_v2):
    def resplit_data(train_meta, num_swap=100000, seed=0, kfold=4):
        print("====Re Split_Data====")
        patient_fold_dict = (
            train_meta[["patient_id", "fold"]]
            .drop_duplicates()
            .set_index("patient_id")["fold"]
            .to_dict()
        )
        patient_ids = list(patient_fold_dict.keys())
        for id_num in range(num_swap):
            np.random.seed(seed + 2 * id_num)
            selected_patients = np.random.choice(patient_ids, size=2, replace=False)
            (
                patient_fold_dict[selected_patients[0]],
                patient_fold_dict[selected_patients[1]],
            ) = (
                patient_fold_dict[selected_patients[1]],
                patient_fold_dict[selected_patients[0]],
            )
        # Update
        train_meta["fold"] = train_meta["patient_id"].map(patient_fold_dict)
        return train_meta

    train_v2 = resplit_data(train_v2)
    return train_v2


def check_irreguler_data(train_csv, eeg_dir, spec_dir):
    # Create a csv file of all rows
    train = pd.read_csv(train_csv)
    TARGETS = train.columns[-6:]

    # train.head(5)
    def get_nan_ratio(data):
        total = data.size
        count_nan = np.count_nonzero(np.isnan(data))
        return count_nan / total

    def all_same(data):
        # return np.nanstd(data) == 0 # eeg_id=1457334423を判別できない(0.16と0で構成されている)ので却下
        return len(np.unique(data)) <= 2

    unique_eeg_id = train["eeg_id"].unique()

    eeg_hz = 200
    spec_hz = 0.5

    train["eeg_nan_ratio"] = 0.0
    train["spec_nan_ratio"] = 0.0
    train["eeg_std_0"] = False

    for eeg_id in tqdm(unique_eeg_id, total=len(unique_eeg_id)):
        rows = train[train["eeg_id"] == eeg_id]
        eeg = pd.read_parquet(eeg_dir / f"{eeg_id}.parquet")
        eeg = eeg.values
        spec_id = rows.iloc[0]["spectrogram_id"]
        spectrogram = pd.read_parquet(spec_dir / f"{spec_id}.parquet")

        for _, row in rows.iterrows():
            eeg_offset = row["eeg_label_offset_seconds"]
            start = int(eeg_offset * eeg_hz)
            eeg_nan_ratio = get_nan_ratio(eeg[start : start + 10_000])
            eeg_std_0 = all_same(eeg[start : start + 10_000])

            spec_offset = row["spectrogram_label_offset_seconds"]
            sub_spec = spectrogram[
                (spectrogram["time"] >= spec_offset)
                & (spectrogram["time"] < spec_offset + 600)
            ]
            sub_spec = sub_spec.values[:, 1:]
            spec_nan_ratio = get_nan_ratio(sub_spec)

            query = (train["eeg_id"] == eeg_id) & (
                train["eeg_label_offset_seconds"] == eeg_offset
            )
            train.loc[query, "eeg_nan_ratio"] = eeg_nan_ratio
            train.loc[query, "eeg_std_0"] = eeg_std_0
            train.loc[query, "spec_nan_ratio"] = spec_nan_ratio
    return train, TARGETS


def merge_fold(train, train_fold, TARGETS):
    train_fold = train_fold[["eeg_id", "fold"]]
    train_fold = pd.merge(train, train_fold, on="eeg_id", how="left")
    y_data = train_fold[TARGETS].values
    train_fold["n_votes"] = y_data.sum(axis=1)
    y_data = y_data / y_data.sum(axis=1, keepdims=True)
    train_fold[TARGETS] = y_data
    # train_fold.head()
    return train_fold


if __name__ == "__main__":
    base_dir = Path("../hms-harmful-brain-activity-classification/")
    output_fold1 = base_dir / "train_fold_irr_mark_v2.csv"
    output_fold2 = base_dir / "train_fold_irr_mark_v5.csv"
    # output_fold1 = base_dir / "train_fold_latesub_v1.csv"
    # output_fold2 = base_dir / "train_fold_latesub_v2.csv"

    train_csv = base_dir / "train.csv"
    eeg_dir = base_dir / "train_eegs"
    spec_dir = base_dir / "train_spectrograms"

    train = group_by_eeg_id(train_csv)
    train_v1 = get_foldv1(train)
    train_v2 = train_v1.copy()
    train_v2 = get_foldv2(train_v2)

    train, TARGETS = check_irreguler_data(
        train_csv,
        eeg_dir,
        spec_dir,
    )
    # print(train.head())
    # print(train_v1.head())
    train_v1 = merge_fold(train, train_v1, TARGETS)
    train_v2 = merge_fold(train, train_v2, TARGETS)

    # print(train_v1.head())
    train_v1.to_csv(output_fold1, index=False)
    train_v2.to_csv(output_fold2, index=False)
