from typing import Tuple

import numpy as np
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


def norm_integral(p):
    # integrate Z_score(p) dp
    return -np.exp(-scipy.special.erfinv(2 * p - 1) ** 2) / (np.pi * 2) ** 0.5


class config:
    """_summary_
    1. global config
        f_size (int): refers to the resolution of the frequency. Defaults to 512.
        f_range (Tuple[float, float]): refers to the frequency range (Hz). Defaults to (20, 17000).
        verbose (bool): If True, make some warnings if need. Defaults to True.
        samplerate (int): Normalize wav file sample rate. Defaults to 48000.
        axis (str): Defines the axis order. F is frequency, T is time, C is channel (Real and Imag). Defaults to "FTC".
        HPI (bool): Frequency rescaled by Human Perception Intensity.  Defaults to False.
        normalize_percentile (Tuple[float, float]): Declare the scope of a sampling group for normalization. Defaults to (0.925, 0.975).

    2. Dataset config
        t_size (int): refers to the time length. Defaults to 1024.
        use_numpy (bool): If True, serve data with numpy array. Otherwise, it is provided as a torch.Tensor . Defaults to True.
        serve_dtype (str): dtype of serving. Defaults to None.
        device (str): only if `use_numpy=False`, configure to where tensor is placed. Defaults to None.

    3. post-processing (revert to wav) config
        wav_dtype (np.dtype): wav file save dtype. Defaults to np.int32.
        Base_HPI_dB (float):global AMP value. Defaults to 87.
        Max_HPI_dB (float): global AMP max value limit. Defaults to 96.
        Max_dB (float): local AMP truncation. Defaults to 92.


    ============================

    Available torch.dtypes:
        https://pytorch.org/docs/stable/tensors.html

        The following are valid cases:
            dtype="float32" # for numpy
            dtype="torch.bfloat16" # for torch
            dtype="torch.float" # for torch

    Available torch device names:
        https://github.com/pytorch/pytorch/blob/7b8cf1f7366bff95e9954037a58a8bb0edaaebd3/c10/core/Device.cpp#L52
    """

    # global config
    f_size: int = 512
    f_range: Tuple[float, float] = (20, 17000)
    verbose: bool = True
    samplerate: int = 48000
    axis: str = "FTC"
    HPI: bool = False
    normalize_percentile: Tuple[float, float] = (0.925, 0.975)

    # Dataset config
    t_size: int = 1024
    use_numpy: bool = True
    serve_dtype: str = None
    device: str = None

    # pfft2wav config
    wav_dtype: np.dtype = np.int32
    Base_HPI_dB: float = 87
    Max_HPI_dB: float = 96
    Max_dB: float = 92

    def __get_freq(self, nperseg):
        frequencies = scipy.fft.rfftfreq(nperseg, 1 / self.samplerate)
        return frequencies

    def __f_index(self, nperseg):
        frequencies = self.__get_freq(nperseg)
        index = [
            i
            for i, I in enumerate(frequencies)
            if self.f_range[0] <= I <= self.f_range[1]
        ]
        return index

    @property
    def nperseg(self):
        def _sub(x):
            return len(self.__f_index(x)) >= self.f_size

        l, r = 1, 2
        while not _sub(r):
            l, r = r, r * 2
        while l < r:
            m = (l + r) // 2
            if _sub(m):
                r = m
            else:
                l = m + 1
        return r

    @property
    def raw_f_size(self):
        # f_size for reconstruct FFT matrix for istft
        return len(self.__get_freq(self.nperseg))

    @property
    def f_index(self):
        return np.asarray(self.__f_index(self.nperseg))

    @property
    def frequencies(self):
        return self.__get_freq(self.nperseg)[np.asarray(self.__f_index(self.nperseg))]

    @property
    def sri(self):
        return get_sri(self.frequencies)

    @property
    def target_norm(self):
        p_low, p_upp = self.normalize_percentile
        assert 0 <= p_low <= 1
        assert 0 <= p_upp <= 1
        return (norm_integral(p_upp) - norm_integral(p_low)) / (p_upp - p_low)

    def __init__(self, **kwargs):
        for K, V in kwargs.items():
            assert hasattr(self, K)
            setattr(self, K, V)

        assert not self.use_numpy or self.device is None
