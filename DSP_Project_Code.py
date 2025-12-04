import numpy as np
import scipy.io as sio
from scipy.signal import iirnotch, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from scipy.stats import skew, kurtosis

# Load EEG Data
mat = sio.loadmat("eeg_data.mat")  
data = mat['data'][:, :4096]       # first 4096 samples per segment
labels = mat['labels'].flatten()   
fs = int(mat['fs'][0,0])           # scalar sampling rate

print("Data shape:", data.shape)
print("Sampling rate:", fs)


def process_50Hz(eeg_segment, fs):
    """
    Apply a 50 Hz notch filter to a single EEG segment
    """
    f0 = 50.0   # frequency to remove
    Q = 30      # quality factor
    b, a = iirnotch(f0, Q, fs)
    filtered = filtfilt(b, a, eeg_segment)
    return filtered

def feature_1():
    pass

def feature_2():
    pass

def feature_3(eeg_segment):
    """
    Compute first-order difference of EEG segment
    """
    return np.diff(eeg_segment, n=1)


def feature_4(eeg_segment):
    derivative = feature_3(eeg_segment)
    return [
        np.mean(derivative),
        np.var(derivative),
        skew(derivative),
        kurtosis(derivative)
    ]

def bandpass_filter(signal, fs, low, high):
    sos = butter(4, [low/(fs/2), high/(fs/2)], btype='band', output='sos')
    return sosfilt(sos, signal)

def feature_5(eeg_segment, fs):
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    features = []
    for low, high in bands.values():
        filtered = bandpass_filter(eeg_segment, fs, low, high)
        features.extend([
            np.mean(filtered),
            np.var(filtered),
            skew(filtered),
            kurtosis(filtered)
        ])
    return features


def classify_80_20():
    pass

def classify_KNN():
    pass

def main():
    # Take the first EEG segment 
    first_segment_raw = data[0, :]

    # Apply 50 Hz notch filter 
    first_segment_filtered = process_50Hz(first_segment_raw, fs)

    # Feature 3: First-order difference 
    feat3 = feature_3(first_segment_filtered)

    #Feature 4: Stats from derivative 
    feat4 = feature_4(first_segment_filtered)

    # Feature 5: Frequency band features
    feat5 = feature_5(first_segment_filtered, fs)

   
    print("Feature 3 (Derivative) length:", len(feat3))
    print("Feature 4 (Derivative stats):", feat4)
    print("Feature 5 (Frequency band features) length:", len(feat5))

    # Plot raw vs filtered EEG segment 
    t = np.arange(0, first_segment_raw.size) / fs  

    plt.figure(figsize=(14, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, first_segment_raw, color='blue', linewidth=0.8)
    plt.title("First EEG Segment (Raw)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")

    plt.subplot(2, 1, 2)
    plt.plot(t, first_segment_filtered, color='orange', linewidth=0.8)
    plt.title("First EEG Segment (After 50 Hz Notch Filter)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
