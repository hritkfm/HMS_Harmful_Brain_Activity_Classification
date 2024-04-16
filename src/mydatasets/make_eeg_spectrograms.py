import os, gc
import cv2
import mne
import pywt
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

NAMES = ["LL", "LP", "RP", "RR"]
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
FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]


def get_mne_raw(df_eeg):
    raw = mne.io.RawArray(df_eeg.to_numpy().T*1e-6, INFO) # µV to V
    return raw

def apply_filter(raw, l_freq=0.5, h_freq=70, notch_freq=60):
    raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq).notch_filter(notch_freq, picks='eeg')
    return raw_filtered

def apply_AER(raw):
    # 平均電極リファレンス：全信号の平均をとり、全信号を平均で引くことでノイズを低減する処理
    raw = raw.copy()
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()
    return raw

def raw2numpy(raw):
    return raw[:][0].T * 1e6


# ここの関数の説明：https://qiita.com/boro1234/items/87472221ba7fd9a07c60
# DENOISE FUNCTION
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet="haar", level=1):
    # wavelet変換。coeffはwavelet係数。"per"は境界の左右端をつなげる
    coeff = pywt.wavedec(x, wavelet, mode="per")

    # 係数の最後(高周波成分)の平均絶対偏差を求める
    sigma = (1 / 0.6745) * maddest(coeff[-level])

    # それを元にスレッシュを決める。
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))

    # threshより小さいものを0にする。
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])

    # ウェーブレット逆変換で戻す。
    ret = pywt.waverec(coeff, wavelet, mode="per")

    return ret


def spectrogram_from_eeg(
    parquet_path,
    eeg=None,
    denoise_wavelet=None,
    start_time=None,
    display=False,
    spec_width=256,
    spec_height = 128,
    hop_length = 10_000 // 256,
    n_fft=1024,
    n_mels=128,
    fmin=0,
    fmax=20,
    win_length=128,
    notch_filtering=False,
    AER_filtering=False,
):
    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    if eeg is None:
        eeg = pd.read_parquet(parquet_path)

    eeg_id = parquet_path.split("/")[-1]
    if start_time is None:
        middle = (len(eeg) - 10_000) // 2
        eeg = eeg.iloc[middle : middle + 10_000]

    else:
        start = start_time * 200
        eeg = eeg.iloc[start : start + 10_000]

    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((spec_height, spec_width, 4), dtype="float32")
    
    if notch_filtering or AER_filtering:
        cols = eeg.columns
        eeg_raw = get_mne_raw(eeg)
        if notch_filtering:
            eeg_raw = apply_filter(eeg_raw)
        if AER_filtering:
            eeg_raw = apply_AER(eeg_raw)
        eeg = raw2numpy(eeg_raw)
        eeg = pd.DataFrame(eeg, columns=cols)

    if display:
        plt.figure(figsize=(10, 7))
    signals = []
    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):
            # COMPUTE PAIR DIFFERENCES
            x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values

            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            # DENOISE
            if denoise_wavelet:
                x = denoise(x, wavelet=denoise_wavelet)
            signals.append(x)

            # RAW SPECTROGRAM
            mel_spec = librosa.feature.melspectrogram(
                y=x,
                sr=200,
                # ウィンドウの移動幅。横解像度を決める。256にしたい場合は、波形の長さ//256とする。
                hop_length=hop_length,
                # FFTを計算する長さ。win_lengthで切り出されたあと、n_fftにゼロパディングされてFFT計算される。長いと周波数解析能力が上がるが、時間解析能力が下がる。win_length以上である必要がある。
                n_fft=n_fft,
                # メルフィルタバンクの使用数。縦解像度を決める。
                n_mels=n_mels,
                fmin=fmin,  # 周波数の最小値
                fmax=fmax,  # 周波数の最大値
                win_length=win_length,  # ウィンドウの長さ。時間分解能に直結する
            )

            # LOG TRANSFORM
            width = (mel_spec.shape[1] // 32) * 32  # 32の倍数に変換する
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[
                :, :width
            ]

            # STANDARDIZE TO -1 TO 1
            mel_spec_db = (mel_spec_db + 40) / 40

            # 縦方向を128にリサイズ
            mel_spec_db = cv2.resize(mel_spec_db, (spec_width, spec_height))
            img[:, :, k] += mel_spec_db

        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:, :, k] /= 4.0

        if display:
            plt.subplot(2, 2, k + 1)
            plt.imshow(img[:, :, k], aspect="auto", origin="lower")
            plt.title(f"EEG {eeg_id} - Spectrogram {NAMES[k]}")

    if display:
        plt.show()
        plt.figure(figsize=(10, 5))
        offset = 0
        for k in range(4):
            if k > 0:
                offset -= signals[3 - k].min()
            plt.plot(range(10_000), signals[k] + offset, label=NAMES[3 - k])
            offset += signals[3 - k].max()
        plt.legend()
        plt.title(f"EEG {eeg_id} Signals")
        plt.show()
        print()
        print("#" * 25)
        print()

    return img
