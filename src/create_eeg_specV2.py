import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from mydatasets.make_eeg_spectrograms import spectrogram_from_eeg
from joblib import Parallel, delayed
from utils.general_utils import tqdm_joblib


os.environ["OMP_NUM_THREADS"] = "1"  # OpenMPで使用するスレッド数を制限
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLASで使用するスレッド数を制限
os.environ["MKL_NUM_THREADS"] = "1"  # MKLで使用するスレッド数を制限
os.environ["NUMBA_NUM_THREADS"] = "1"

def create_eeg_spec(row):
    eeg_id = row.eeg_id
    eeg_offset_list = row.eeg_label_offset_seconds
    # CREATE SPECTROGRAM FROM EEG PARQUET
    for offset in eeg_offset_list:
        if parquet_type == "all":
            img = spectrogram_from_eeg(
                f"{PATH}{eeg_id}.parquet",
                start_time=int(offset),
                **params
            )
        else:
            img = spectrogram_from_eeg(
                f"{PATH}{eeg_id}_{int(offset)}.parquet",
                start_time=0,
                **params
            )
        # SAVE TO DISK
        np.save(f"{directory_path}{eeg_id}_{int(offset)}.npy", img)
        # all_eegs[eeg_id] = img  


# メルスペクトログラムのパラメータを変えてデータを作成
if __name__ == "__main__":
    # =========ver2_db8=============
    # ver = "ver2_db8"
    # params = {
    #     "denoise_wavelet": "db8",
    #     "spec_width": 256,
    #     "spec_height": 128,
    #     "hop_length": 10_000 // 256,
    #     "n_fft": 1024,
    #     "n_mels": 128,
    #     "fmin": 0,
    #     "fmax": 20,
    #     "win_length": 128,
    #     "notch_filtering": False,
    #     "AER_filtering": False,
    # }
    # =========ver2_haar=============
    ver = "ver2_haar"
    params = {
        "denoise_wavelet": "haar",
        "spec_width": 256,
        "spec_height": 128,
        "hop_length": 10_000 // 256,
        "n_fft": 1024,
        "n_mels": 128,
        "fmin": 0,
        "fmax": 20,
        "win_length": 128,
        "notch_filtering": False,
        "AER_filtering": False,
    }
    # ===============================

    # =======ver3_filtered===========
    # ver = "ver3_filtered"
    # params = {
    #     "denoise_wavelet": None,
    #     "spec_width": 256,
    #     "spec_height": 128,
    #     "hop_length": 10_000 // 256,
    #     "n_fft": 1024,
    #     "n_mels": 128,
    #     "fmin": 0,
    #     "fmax": 20,
    #     "win_length": 128,
    #     "notch_filtering": True,
    #     "AER_filtering": True,
    # }
    # ===============================
    # =============ver4===============
    # ver = "ver4"
    # params = {
    #     "denoise_wavelet": None,
    #     "spec_width": 512,
    #     "spec_height": 256,
    #     "hop_length": 10_000 // 512,
    #     "n_fft": 1024,
    #     "n_mels": 192,
    #     "fmin": 0,
    #     "fmax": 20,
    #     "win_length": 128,
    #     "notch_filtering": False,
    #     "AER_filtering": False,
    # }
    # ===============================
    
    train = pd.read_csv("../hms-harmful-brain-activity-classification/train.csv")
    PATH = "../hms-harmful-brain-activity-classification/train_eegs/"
    # PATH = "../hms-harmful-brain-activity-classification/train_eegs_filtered/ver1/"
    parquet_type = ["all" ,"per_offset"][0] # ファイルがすでにオフセットごとに分割されている場合は1

    directory_path = (
        f"../hms-harmful-brain-activity-classification/EEG_Spectrograms/{ver}/"
    )
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # all_eegs = {}

    print(f"Creating and writing {len(train)} spectrograms to disk... ", end="")

    with tqdm_joblib(total=len(train["eeg_id"].unique())):
        Parallel(n_jobs=20)(delayed(create_eeg_spec)(
            row
            ) for _, row in train.groupby("eeg_id")["eeg_label_offset_seconds"]
            .apply(list)
            .reset_index()
            .iterrows()
            )

    # train.csvのeeg_idごとにループ.eeg_label_offsetをリストとして取得し、
    # train.csvの各行のeeg_specをoffset分ずらして保存

    


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
    #     # CREATE SPECTROGRAM FROM EEG PARQUET
    #     for offset in eeg_offset_list:
    #         img = spectrogram_from_eeg(
    #             f"{PATH}{eeg_id}.parquet",
    #             denoise_wavelet=USE_WAVELET,
    #             start_time=int(offset),
    #         )
    #         # SAVE TO DISK
    #         np.save(f"{directory_path}{eeg_id}_{int(offset)}.npy", img)
    #         # all_eegs[eeg_id] = img

    # # SAVE EEG SPECTROGRAM DICTIONARY
    # # np.save(f"{directory_path}all.npy", all_eegs)
