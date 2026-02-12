from scipy.signal import butter, filtfilt
from scipy.fft import fft
import numpy as np
import pandas as pd

# Preprocessing Denoising: Apply band-pass filter to isolate key frequency range (e.g., 10–2000 Hz).
def bandpass_filter(signal, fs, lowcut=10, highcut=2000, order=4):
    """
    Apply a Butterworth bandpass filter to a signal.
    
    Parameters:
    - signal: Input time-series signal (1D array)
    - fs: Sampling frequency in Hz
    - lowcut: Lower cutoff frequency (Hz)
    - highcut: Upper cutoff frequency (Hz)
    - order: Filter order
    
    Returns:
    - Filtered signal (1D array)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Segmentation: Split long time-series into smaller overlapping windows (e.g., 1024 samples with 50% overlap).
def segment_signal(signal, window_size=1024, overlap=0.5):
    """
    Split a 1D signal into overlapping windows.

    Parameters:
    - signal: 1D NumPy array (the time-series signal)
    - window_size: Number of samples in each window
    - overlap: Fraction of overlap between windows (e.g., 0.5 for 50%)

    Returns:
    - segments: 2D NumPy array of shape (num_windows, window_size)
    """
    step = int(window_size * (1 - overlap))
    num_windows = ((len(signal) - window_size) // step) + 1

    segments = np.array([
        signal[i * step : i * step + window_size]
        for i in range(num_windows)
    ])
    return segments

# Normalization: Normalize each window (mean = 0, std = 1).
def normalize_windows(windows):
    """
    Normalize each window in a 2D array to have mean 0 and std 1.

    Parameters:
    - windows: 2D NumPy array of shape (num_windows, window_size)

    Returns:
    - normalized_windows: Same shape, each row normalized
    """
    means = np.mean(windows, axis=1, keepdims=True)
    stds = np.std(windows, axis=1, keepdims=True)
    
    # Avoid division by zero
    stds[stds == 0] = 1.0
    
    normalized = (windows - means) / stds
    return normalized

# Frequency Domain Transformation
def compute_fft_features(signal, sampling_rate):
    N = len(signal)
    fft_vals = np.abs(fft(signal))[:N//2]
    freqs = np.fft.fftfreq(N, 1/sampling_rate)[:N//2]
    return freqs, fft_vals

#Feature Extraction (from FFT) Instead of using the entire FFT, extract compact and informative features like:
#Feature Description Peak Frequency - Frequency with maximum amplitude
#Spectral Centroid - Weighted mean frequency
#Spectral Kurtosis - Indicates impulsiveness
#Band Power (in bands) - Sum of energy in different frequency bands
#Harmonic Energy Ratio - Ratio of harmonic to total energy
def extract_fft_features(freqs, fft_vals):
    peak_freq = freqs[np.argmax(fft_vals)]
    centroid = np.sum(freqs * fft_vals) / np.sum(fft_vals)
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * fft_vals) / np.sum(fft_vals))
    return [peak_freq, centroid, bandwidth]

def rdg(df, State=None, sensor=None):
    df_st = df[df.State==State] if State is not None else df
    df_se = df_st[df_st.sensor==sensor] if sensor is not None else df_st
    return df_se

