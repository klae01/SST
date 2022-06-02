import math
import warnings
from collections import Counter

import cv2
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal

from .config import config as SST_config


def complex_to_inner(data: np.ndarray):
    if data.dtype == np.complex64:
        return data.view("(2,)float32")
    if data.dtype == np.complex128:
        return data.view("(2,)float64")
    raise NotImplementedError(f"No implementation for {data.dtype}")


def inner_to_complex(data: np.ndarray):
    if data.dtype == np.float32:
        return data.view("complex64").squeeze(-1)
    if data.dtype == np.float64:
        return data.view("complex128").squeeze(-1)
    raise NotImplementedError(f"No implementation for {data.dtype}")


def __axis_inner(data: np.ndarray, axis: str):
    # Reverts from the outer axis order specified in the config to the inner axis order.
    assert Counter("FTC") == Counter(axis)
    axis_order = dict(map(reversed, enumerate(axis)))
    axis_order = [axis_order[I] for I in "FTC"]
    return data.transpose(axis_order)


def __axis_outer(data: np.ndarray, axis: str):
    # Convert to the axis order required by config from inner axis order.
    assert Counter("FTC") == Counter(axis)
    axis_order = dict(map(reversed, enumerate("FTC")))
    axis_order = [axis_order[I] for I in axis]
    return data.transpose(axis_order)


def wav2pfft(file: str, config: SST_config):
    samplerate, samples = wav.read(file)
    if len(samples.shape) > 1:
        samples = samples[..., 0]
    samples = scipy.signal.resample_poly(
        samples.astype(np.float32), config.samplerate, samplerate
    )
    samplerate = config.samplerate

    frequencies, times, spectrogram = scipy.signal.stft(
        samples, samplerate, nperseg=config.nperseg
    )
    spectrogram = spectrogram[config.f_index, ...]

    if config.HPI:
        spectrogram *= config.sri

    norm = abs(spectrogram).sum(axis=0)
    idx_low, idx_upp = [
        method(len(norm) * x)
        for method, x in zip([math.floor, math.ceil], config.normalize_percentile)
    ]
    current_norm = np.partition(norm, [idx_low, idx_upp])[idx_low : idx_upp + 1].mean()

    spectrogram *= config.target_norm / current_norm
    spectrogram = complex_to_inner(spectrogram)
    spectrogram = __axis_outer(spectrogram, axis=config.axis)
    return spectrogram


def pfft2wav(spectrogram: np.ndarray, config: SST_config):
    spectrogram = __axis_inner(spectrogram, axis=config.axis)
    norm = np.linalg.norm(spectrogram, ord=2, axis=-1)

    if config.HPI:
        sri = norm.sum(axis=0)
        spectrogram /= config.sri[:, None]
    else:
        sri = np.einsum("f,ft->t", config.sri, norm)

    idx_low, idx_upp = [
        method(len(norm) * x)
        for method, x in zip([math.floor, math.ceil], config.normalize_percentile)
    ]
    sri_current_norm = np.partition(sri, [idx_low, idx_upp])[
        idx_low : idx_upp + 1
    ].mean()
    del sri

    GLOBAL_AMP = np.log10(sri_current_norm / config.target_norm + 1e-9) * 10
    GLOBAL_AMP = min(config.Base_HPI_dB, config.Max_HPI_dB - GLOBAL_AMP)

    LOCAL_MAX_AMP = config.Max_dB - np.log10(norm.sum(axis=0) + 1e-9) * 10
    AMP = np.minimum(GLOBAL_AMP, LOCAL_MAX_AMP)

    if config.verbose:
        OVER_AMP = np.mean(GLOBAL_AMP > LOCAL_MAX_AMP)
        if OVER_AMP:
            warnings.warn(
                f"{OVER_AMP*100: .2f}% of spectrogram has boosted unexpected."
                " Automatically clip the intensity."
            )

    spectrogram = spectrogram * 10 ** (AMP[None, :, None] / 10)
    padded_spectrogram = np.zeros(
        (config.raw_f_size, *spectrogram.shape[1:]), dtype=spectrogram.dtype
    )
    padded_spectrogram[config.f_index] = spectrogram

    r_times, r_samples = scipy.signal.istft(
        inner_to_complex(padded_spectrogram), config.samplerate
    )

    info = np.iinfo(config.wav_dtype)
    data = np.clip(r_samples, info.min, info.max)
    data = data.astype(config.wav_dtype)
    return data


def pfft2img(spectrogram: np.ndarray, config: SST_config):
    def as_uint8(X):
        return np.clip(X * 256, 0, 255).astype(np.uint8)

    img = __axis_inner(spectrogram, axis=config.axis)
    scale = np.linalg.norm(img, ord=2, axis=-1, keepdims=True)

    AMP = scale / (scale.max() + 1e-6)
    img_ab = img * (AMP / (scale + 1e-6))
    img = np.concatenate([AMP, (img_ab + 1) / 2], axis=-1)
    img = as_uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return img


def limit_length_img(img):
    assert img.dtype == np.uint8
    IMG = img
    IMG = np.array_split(IMG, range(2048, IMG.shape[1], 2048), axis=-2)
    LAST = np.zeros_like(IMG[0])
    LAST[:, : IMG[-1].shape[1]] = IMG[-1][...]
    IMG[-1] = LAST
    IMG = np.concatenate(IMG, axis=0)
    return IMG
