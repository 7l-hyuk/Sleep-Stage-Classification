import pandas as pd
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis

eeg_bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 45)
        }


def band_pass_filter(segment: pd.Series) -> list[np.ndarray]:
    filtered_features = []
    for band in eeg_bands:
        low, high = eeg_bands[band]
        filter = signal.butter(
            3, [low, high],
            btype='bandpass',
            fs=100,
            output='sos'
            )
        filtered = signal.sosfilt(filter, segment)
        filtered_features += [filtered]
    return filtered_features


def esis(segment: np.ndarray, band_name: str, gamma: int = 100) -> float:
    velocity = dict()
    for band in eeg_bands:
        velocity[band] = (sum(eeg_bands[band])/2)*gamma
    return np.sum((segment)**2)*velocity[band_name]


def mmd(
        segment: np.ndarray,
        window_size: int = 100
        ) -> float:
    mmd_val = 0
    for i in range(len(segment)//window_size):
        window = segment[i*window_size: (i+1)*window_size]
        max_point = None
        min_point = None
        max_val = np.max(window)
        min_val = np.min(window)

        for x in range(len(window)):
            if window[x] == max_val:
                max_point = np.array([x, max_val])
                break

        for x in range(len(window)):
            if window[x] == min_val:
                min_point = np.array([x, min_val])
                break
        mmd_val += (np.sum((max_point - min_point)**2))**(1/2)
    return mmd_val


def fourier_transformed_stats_values(
        segment: np.ndarray,
        sample_rate: float = 100.0,
        low_freq: float = None,
        high_freq: float = None
        ) -> tuple[np.ndarray]:
    n = len(segment)
    yf = fft(segment)
    if (low_freq is not None) and (high_freq is not None):
        xf = fftfreq(n, 1 / sample_rate)
        freq_mask = (xf[:n // 2] >= low_freq) & (xf[:n // 2] <= high_freq)
        amplitude = np.abs(yf[:n // 2])
        amplitude = amplitude[freq_mask]
    else:
        amplitude = np.abs(yf[:n // 2])
    return (
        np.mean(amplitude),
        np.median(amplitude),
        np.min(amplitude),
        np.max(amplitude),
        np.std(amplitude),
        skew(amplitude),
        kurtosis(amplitude)
        )
