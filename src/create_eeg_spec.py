import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from mydatasets.make_eeg_spectrograms import spectrogram_from_eeg


# メルスペクトログラムのパラメータを変えてデータを作成
if __name__ == "__main__":
    ver = ""
    USE_WAVELET = None  # "db8"  # None  # or "db8" or anything below
    win_length = 512
    n_mels = 32
    DISPLAY = 0

    train = pd.read_csv("../hms-harmful-brain-activity-classification/train.csv")
    PATH = "../hms-harmful-brain-activity-classification/train_eegs/"

    wavelet = f"_{USE_WAVELET}" if USE_WAVELET is not None else ""

    directory_path = f"../hms-harmful-brain-activity-classification/EEG_Spectrograms{ver}_{win_length}_{n_mels}{wavelet}/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    EEG_IDS = train.eeg_id.unique()
    all_eegs = {}

    for i, eeg_id in tqdm(enumerate(EEG_IDS), total=len(EEG_IDS)):
        # CREATE SPECTROGRAM FROM EEG PARQUET
        img = spectrogram_from_eeg(
            f"{PATH}{eeg_id}.parquet",
            denoise_wavelet=USE_WAVELET,
            n_mels=n_mels,
            win_length=win_length,
        )

        # SAVE TO DISK
        if i == DISPLAY:
            print(
                f"Creating and writing {len(EEG_IDS)} spectrograms to disk... ", end=""
            )
        np.save(f"{directory_path}{eeg_id}", img)
        all_eegs[eeg_id] = img

    # SAVE EEG SPECTROGRAM DICTIONARY
    np.save(f"{directory_path}all.npy", all_eegs)
