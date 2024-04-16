import numpy as np
import colorednoise as cn

def PinkNoiseSNR(samples, sample_rate, min_snr=10, max_snr=40):
    # audiomentationsのlambda関数に入れる用
    snr = np.random.uniform(min_snr, max_snr)
    a_signal = np.sqrt(samples**2).max()
    a_noise = a_signal / (10 ** (snr / 20))
    pink_noise = cn.powerlaw_psd_gaussian(1, samples.shape[-1])[np.newaxis, :]
    # pink_noise = cn.powerlaw_psd_gaussian(1, samples.shape)
    a_pink = np.sqrt(pink_noise**2).max()
    augmented = (samples + pink_noise * 1 / a_pink * a_noise).astype(samples.dtype)
    return augmented


def ChannelSwap(samples, sample_rate, type="half"):
    # audiomentationsのlambda関数に入れる用
    # channel-first (https://iver56.github.io/audiomentations/guides/multichannel_audio_array_shapes/)
    # half:チャネルの前半分と後半分を入れ替える。
    # reverse:チャネルの順番を逆順にする。
    # random: チャネルの順番をランダムにする。
    if type == "half":
        mid = len(samples) // 2
        samples = np.concatenate((samples[mid:], samples[:mid]))
    elif type == "reverse":
        samples = samples[::-1]
    elif type == "random":
        np.random.shuffle(samples)
    return samples

def ChannelShuffle(samples, sample_rate):
    mid = len(samples) // 2
    first = samples[:mid]
    last = samples[mid:]
    np.random.shuffle(first)
    np.random.shuffle(last)
    return np.concatenate((first, last))
