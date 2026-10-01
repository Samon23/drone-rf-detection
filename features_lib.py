"""The feature function, kept in one place so training and the noise test use identical code."""
import numpy as np
from scipy import signal
from common import FS

N_BANDS = 64   # squeeze each 513-point spectrum into 64 frequency bands


def band_features(x):
    """Log power in N_BANDS frequency bands for one short window."""
    f, pxx = signal.welch(x, fs=FS, nperseg=1024)
    bands = [chunk.mean() for chunk in np.array_split(pxx, N_BANDS)]
    return np.log10(np.array(bands) + 1e-12)


def band_centres_mhz():
    """Centre frequency of each band in MHz, in the same order as band_features."""
    f = np.fft.rfftfreq(1024, d=1 / FS)
    return np.array([chunk.mean() for chunk in np.array_split(f, N_BANDS)]) / 1e6
