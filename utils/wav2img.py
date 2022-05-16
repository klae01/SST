import warnings

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal


def log_fit(X, dtype):
    info = np.iinfo(dtype)
    conv = np.sign(X) * np.log2(abs(X) + 1)
    conv = conv / 32
    if conv.min() < -1 or conv.max() > 1:
        warnings.warn(
            f"log_fit: The range of frequency strength [{conv.min(): .5f}, {conv.max(): .5f}] "
            + f"exceeds expectations [-1, 1]."
        )
        conv = np.clip(conv, -1, 1)
    conv = (info.max - info.min) * (conv + 1) / 2 - info.min
    conv = np.clip(conv, info.min, info.max)
    return conv.astype(dtype)


def log_inv(X, dtype):
    info = np.iinfo(dtype)
    if X.dtype not in [np.float32, np.float64]:
        X = X.astype(np.float32)
    X = (X + info.min) * 2 / (info.max - info.min) - 1
    X = X * 32
    X = np.sign(X) * (2 ** abs(X) - 1)
    return X


def wav2img(file, dtype=np.uint8, options={"nperseg": 511}, virtual_samplerate=None):
    samplerate, samples = wav.read(file)
    if virtual_samplerate is not None and virtual_samplerate != samplerate:
        samples = scipy.signal.resample(
            samples, len(samples) * virtual_samplerate // samplerate
        )
        samplerate = virtual_samplerate
    frequencies, times, spectrogram = scipy.signal.stft(samples, samplerate, **options)
    spectrogram = np.stack(
        [log_fit(I, dtype) for I in [spectrogram.real, spectrogram.imag]], axis=-1
    )
    return spectrogram, samplerate, dtype


def img2wav(spectrogram: np.ndarray, samplerate: int, dtype: np.dtype):
    spectrogram = log_inv(spectrogram, dtype)
    spectrogram = spectrogram[..., 0] + 1j * spectrogram[..., 1]
    r_times, r_samples = scipy.signal.istft(spectrogram, samplerate)

    info = np.iinfo(np.int32)
    if r_samples.min() < info.min or r_samples.max() > info.max:
        warnings.warn(
            f"img2wav: The range of sampling [{r_samples.min(): .5f}, {r_samples.max(): .5f}] "
            + f"exceeds expectations [{info.min}, {info.max}]."
        )

    return r_samples.astype(np.int32)


try:
    import cv2

    def limit_length_img(img):
        assert img.dtype == np.uint8
        LAB = np.concatenate(
            [np.ones(img.shape[:2] + (1,), np.uint8) * 127, img], axis=-1
        )
        IMG = cv2.cvtColor(LAB, cv2.COLOR_LAB2RGB)
        IMG = np.array_split(IMG, range(2048, IMG.shape[1], 2048), axis=1)
        LAST = np.zeros_like(IMG[0])
        LAST[:, : IMG[-1].shape[1]] = IMG[-1][...]
        IMG[-1] = LAST
        IMG = np.concatenate(IMG, axis=0)
        return IMG

except:
    ...
