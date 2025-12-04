import numpy as np
import scipy.io as sio
from scipy.signal import iirnotch, filtfilt
import matplotlib.pyplot as plt

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

def feature_3():
    pass

def feature_4():
    pass

def feature_5():
    pass

def classify_80_20():
    pass

def classify_KNN():
    pass

def main():
    # Apply notch filter to the first segment
    first_segment_raw = data[0, :]
    first_segment_filtered = process_50Hz(first_segment_raw, fs)
    
    # Plot raw vs filtered
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
