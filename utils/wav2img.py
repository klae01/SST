import functools
import warnings

import cv2
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import scipy.special

iso226_base = 40
iso226_freq = np.array([
           20. ,    25. ,    31.5,    40. ,    50. ,    63. ,    80. ,
          100. ,   125. ,   160. ,   200. ,   250. ,   315. ,   400. ,
          500. ,   630. ,   800. ,  1000. ,  1250. ,  1600. ,  2000. ,
         2500. ,  3150. ,  4000. ,  5000. ,  6300. ,  8000. , 10000. ,
        12500. , 20000. ])
iso226_espl = np.array([
        99.85392334, 93.94441144, 88.16590253, 82.62867609, 77.78487094,
        73.08254532, 68.47788682, 64.37114939, 60.58550325, 56.70224677,
        53.40873978, 50.3992414 , 47.5774866 , 44.97662259, 43.05067937,
        41.339195  , 40.06176083, 40.01004637, 41.81945508, 42.50756876,
        39.2296391 , 36.50900986, 35.60891914, 36.64917709, 40.00774113,
        45.82828132, 51.79680693, 54.28413025, 51.48590719, 99.85392334])
        

@functools.lru_cache(maxsize=4)
def get_sri(frequencies):  # sound recognition intensity
    upp_index = np.minimum(
        np.searchsorted(iso226_freq, frequencies, "left"), len(iso226_freq) - 1
    )
    low_index = np.where(upp_index == 0, 1, upp_index - 1)
    sri = (
        iso226_espl[upp_index] * (frequencies - iso226_freq[low_index])
        + iso226_espl[low_index] * (iso226_freq[upp_index] - frequencies)
    ) / (iso226_freq[upp_index] - iso226_freq[low_index])
    sri = 10 ** ((iso226_base - sri) / 10)
    return sri


@functools.lru_cache(maxsize=4)
def norm_integral(p):
    # integrate Z_score(p) dp
    return -np.exp(-scipy.special.erfinv(2 * p - 1) ** 2) / (np.pi * 2) ** 0.5


def wav2pfft(
    file,
    virtual_samplerate=None,
    stft_option={"nperseg": 511},
    normalize_target_option={"p_low": 0.075, "p_upp": 0.025, "HPI": True},
):

    assert 0 <= normalize_target_option["p_low"] <= 1
    assert 0 <= normalize_target_option["p_upp"] <= 1
    samplerate, samples = wav.read(file)
    if virtual_samplerate is not None and virtual_samplerate != samplerate:
        samples = scipy.signal.resample(
            samples, len(samples) * virtual_samplerate // samplerate
        )
        samplerate = virtual_samplerate
    frequencies, times, spectrogram = scipy.signal.stft(
        samples, samplerate, **stft_option
    )

    p_low, p_upp = [1 - normalize_target_option[x] for x in ["p_low", "p_upp"]]
    target_norm = norm_integral(p_upp) - norm_integral(p_low) / (p_upp - p_low)

    if normalize_target_option["HPI"]:
        spectrogram *= get_sri(tuple(frequencies))[..., None]
        HPI = abs(spectrogram).sum(axis=0)
    else:
        HPI = (abs(spectrogram) * get_sri(tuple(frequencies))[..., None]).sum(axis=0)
    idx_low, idx_upp = [int(len(HPI) * x) for x in [p_low, p_upp]]
    current_norm = np.partition(HPI, [idx_low, idx_upp])[idx_low : idx_upp + 1].mean(
        axis=-1
    )

    spectrogram *= target_norm / current_norm  # normalize volume
    spectrogram = np.stack([spectrogram.real, spectrogram.imag], axis=-1)
    return spectrogram, samplerate


def pfft2wav(
    spectrogram: np.ndarray,
    samplerate: int,
    dtype: np.dtype = np.int32,
    normalize_target_option={"AMP": 73, "HPI": True},
):
    # assume spectrogram is well normalize
    # it means, overall average Human Perceptual Intensity set 0dB

    info = np.iinfo(dtype)
    # spectrogram = spectrogram-spectrogram.mean(axis = (0, 1), keepdims = True)
    spectrogram = spectrogram[..., 0] + 1j * spectrogram[..., 1]
    if normalize_target_option["HPI"]:
        frequencies = scipy.fft.rfftfreq(spectrogram.shape[0] * 2 - 1, 1 / samplerate)
        spectrogram /= get_sri(tuple(frequencies))[..., None]
    r_times, r_samples = scipy.signal.istft(spectrogram, samplerate)
    r_samples *= 10 ** (normalize_target_option["AMP"] / 10)
    data = np.clip(r_samples, info.min, info.max)
    data = data.astype(dtype)
    return data


def pfft2img(spectrogram: np.ndarray):
    def as_uint8(X):
        return np.clip(X * 255 + 0.5, 0, 255).astype(np.uint8)

    # img = spectrogram-spectrogram.mean(axis = (0, 1), keepdims = True)
    img = spectrogram
    scale = np.linalg.norm(img, ord=2, axis=-1)
    scale /= scale.max()

    rate = img / abs(img).sum(axis=-1, keepdims=True)
    img = rate * scale[..., None]
    img = as_uint8((img + 1) / 2)
    img = np.concatenate([as_uint8(scale)[..., None], img], axis=-1)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return img


def limit_length_img(img):
    assert img.dtype == np.uint8
    IMG = img
    IMG = np.array_split(IMG, range(2048, IMG.shape[1], 2048), axis=1)
    LAST = np.zeros_like(IMG[0])
    LAST[:, : IMG[-1].shape[1]] = IMG[-1][...]
    IMG[-1] = LAST
    IMG = np.concatenate(IMG, axis=0)
    return IMG
