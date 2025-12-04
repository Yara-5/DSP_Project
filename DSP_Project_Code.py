import numpy as np
import scipy.io as sio
from scipy.signal import iirnotch, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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


def classify_80_20(data, labels, fs):
    """
    Perform 80-20 train/test split and classify using KNN.
    Compare three representations:
    1) Time-domain raw EEG
    2) Derivative stats
    3) Frequency-band features
    """
    X_raw = []
    X_deriv = []
    X_band = []

    for segment in data:
        filtered = process_50Hz(segment, fs)

        # 1) Raw signal
        X_raw.append(filtered)

        # 2) Derivative stats
        X_deriv.append(feature_4(filtered))

        # 3) Frequency-band features
        X_band.append(feature_5(filtered, fs))

    # Convert to arrays
    X_raw = np.array(X_raw)
    X_deriv = np.array(X_deriv)
    X_band = np.array(X_band)
    y = np.array(labels)

    representations = {
        "Raw EEG": X_raw,
        "Derivative stats": X_deriv,
        "Frequency-band features": X_band
    }

    for name, X in representations.items():
        print(f"\nClassification using {name}:")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        for k in range(1, 11):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"K={k}, Accuracy={acc:.3f}")


def classify_KNN():
    pass
def plot_average_spectrum(data, labels, fs):
    """
    Plot the average spectrum for each class.
    """
    classes = np.unique(labels)
    plt.figure(figsize=(12, 6))
    
    for cls in classes:
        class_segments = data[labels == cls]  # select segments of this class
        spectra = []
        
        for seg in class_segments:
            # Apply 50 Hz notch filter
            filtered = process_50Hz(seg, fs)
            
            # Apply Hanning window
            windowed = filtered * np.hanning(len(filtered))
            
            # FFT
            fft_vals = np.fft.rfft(windowed)
            fft_power = np.abs(fft_vals) ** 2
            
            spectra.append(fft_power)
        
        # Average across all segments in this class
        avg_spectrum = np.mean(spectra, axis=0)
        freqs = np.fft.rfftfreq(len(filtered), d=1/fs)
        
        plt.plot(freqs, avg_spectrum, label=f"Class {cls} (n={len(class_segments)})")
    
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title("Average EEG Spectrum per Class")
    plt.legend()
    plt.xlim(0, 60)  # focus on 0-60 Hz range
    plt.show()

def inspect_data(data, labels, fs, num_segments=3):
    """
    Plot a few EEG segments from each class to inspect the data.
    """
    classes = np.unique(labels)
    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        print(f"Class {cls}: {len(cls_indices)} segments")
        
        plt.figure(figsize=(14, 4 * num_segments))
        for i, idx in enumerate(cls_indices[:num_segments]):
            segment = process_50Hz(data[idx], fs)
            t = np.arange(0, len(segment)) / fs
            plt.subplot(num_segments, 1, i+1)
            plt.plot(t, segment, color='orange', linewidth=0.8)
            plt.title(f"Class {cls} EEG Segment {i+1} (Filtered)")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude (µV)")
        plt.tight_layout()
        plt.show()

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
    plot_average_spectrum(data, labels, fs)

    inspect_data(data, labels, fs, num_segments=3)

    # Run 80-20 classification 
    classify_80_20(data, labels, fs)

if __name__ == "__main__":
    main()
