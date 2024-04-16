import os, gc
import cv2
import pywt
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
import joblib
from pathlib import Path
from mne.preprocessing import ICA
from tqdm import tqdm
from joblib import Parallel, delayed
from utils.general_utils import tqdm_joblib

mne.set_log_level(verbose=False)
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


INFO = mne.create_info(
    FEATURE_LIST,
    sfreq=200,
    ch_types=(["eeg"] * (len(FEATURE_LIST) -1)) + ["ecg"]
)
INFO.set_montage("standard_1020")

def get_mne_raw(df_eeg):
    raw = mne.io.RawArray(df_eeg.to_numpy().T*1e-6, INFO) # µV to V
    return raw

def apply_filter(raw, l_freq=0.5, h_freq=70, notch_freq=60):
    raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq).notch_filter(notch_freq, picks='eeg')
    return raw_filtered

def apply_ica(raw):
    ica_raw = raw.copy()
    ecg_events = mne.preprocessing.find_ecg_events(ica_raw)

    # ICAの実行
    # ica = ICA(n_components=None, random_state=97, max_iter=800)
    ica = ICA(n_components=None, random_state=97, max_iter="auto")
    ica.fit(ica_raw)

    # EKGに基づくアーティファクト成分の特定
    ecg_indices, ecg_scores = ica.find_bads_ecg(ica_raw, method='ctps')

    # アーティファクト成分の除去
    ica.exclude = ecg_indices
    ica.apply(ica_raw)
    return ica_raw


def apply_AER(raw):
    # 平均電極リファレンス：全信号の平均をとり、全信号を平均で引くことでノイズを低減する処理
    raw = raw.copy()
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()
    return raw

def raw2numpy(raw):
    return raw[:][0].T * 1e6

def mne_preprocessing(df_eeg):
    raw = get_mne_raw(df_eeg)
    raw = apply_filter(raw, 0.5, 70, 60)
    # raw = apply_ica(raw)
    raw = apply_AER(raw)
    eeg = raw2numpy(raw)
    return eeg
        

def make_filtered_eeg(row, eeg_dir, output_dir):
    mne.set_log_level(verbose=False)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    eeg_id = row.eeg_id
    eeg_offset_list = row.eeg_label_offset_seconds
    eeg = pd.read_parquet(f"{eeg_dir}/{eeg_id}.parquet")
    for offset in eeg_offset_list:
        start = int(offset * 200)
        eeg_subset = eeg.iloc[start: start + 10_000]
        try:
            filtered_eeg_subset = mne_preprocessing(eeg_subset)
            # train.loc[(train["eeg_id"] == eeg_id) & (train["eeg_label_offset_seconds"] == offset), "filtering"] = 1
            if np.isnan(filtered_eeg_subset).sum() > 0: # NANが一つでもある場合はそのまま使用
                filtered_eeg_subset = eeg_subset.values
        except:
            print("error:", eeg_id, offset)
            filtered_eeg_subset = eeg_subset.values
        filtered_eeg_subset = pd.DataFrame(columns=FEATURE_LIST, data=filtered_eeg_subset)
        save_path = f"{output_dir}/{eeg_id}_{int(offset)}.parquet"
        filtered_eeg_subset.to_parquet(save_path, index=False)

# EEGデータの前処理をして保存
if __name__ == "__main__":
    output_dir = "../hms-harmful-brain-activity-classification/train_eegs_filtered/ver1"
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    train = pd.read_csv("../hms-harmful-brain-activity-classification/train.csv")
    eeg_dir = "../hms-harmful-brain-activity-classification/train_eegs"

    # train.csvのeeg_idごとにループ.eeg_label_offsetをリストとして取得し、
    # train.csvの各行のeeg_specをoffset分ずらして保存

    train["filtering"] = 0
    # train = train.iloc[:10]
    total = len(train["eeg_id"].unique())    
    with tqdm_joblib(total=total):
        joblib.Parallel(n_jobs=10)(
            joblib.delayed(make_filtered_eeg)(
                row, eeg_dir, output_dir
                ) for _, row in train.groupby("eeg_id")["eeg_label_offset_seconds"]
                .apply(list)
                .reset_index()
                .iterrows())
    # train.to_csv("../hms-harmful-brain-activity-classification/train_filtered.csv")

    # for _, row in tqdm(
    #     (
    #         train.groupby("eeg_id")["eeg_label_offset_seconds"]
    #         .apply(list)
    #         .reset_index()
    #         .iterrows()
    #     ),
    #     total=len(train.eeg_id.unique()),
    # ):
    #     eeg_id = row.eeg_id
    #     eeg_offset_list = row.eeg_label_offset_seconds
    #     eeg = pd.read_parquet(f"{eeg_dir}/{eeg_id}.parquet")
    #     for offset in eeg_offset_list:
    #         start = int(offset * 200)
    #         eeg_subset = eeg.iloc[start: start + 10_000]
    #         try:
    #             filtered_eeg_subset = mne_preprocessing(eeg_subset)
    #             train.loc[(train["eeg_id"] == eeg_id) & (train["eeg_label_offset_seconds"] == offset), "filtering"] = 1
    #         except:
    #             print("error:", eeg_id, offset)
    #             filtered_eeg_subset = eeg_subset.values
    #         filtered_eeg_subset = pd.DataFrame(columns=FEATURE_LIST, data=filtered_eeg_subset)
    #         save_path = f"{output_dir}/{eeg_id}_{int(offset)}.parquet"
    #         filtered_eeg_subset.to_parquet(save_path, index=False)
    


