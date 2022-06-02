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
    # Dataset config
    f_size: int = 512
    t_size: int = 1024
    f_range: Tuple[float, float] = [20, 17000]  # term in Hz

    use_numpy: bool = True
    return_dtype: str = None
    device: str = None

    # pre/post processing config
    samplerate: int = 48000
    axis: str = "FTC"
    HPI: bool = False

    normalize_percentile: Tuple[float, float] = (0.925, 0.975)

    # pfft2wav config
    Base_HPI_dB: float = 93  # global AMP value
    Max_HPI_dB: float = 102  # global AMP max value limit
    Max_dB: float = 96  # local AMP truncation

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
