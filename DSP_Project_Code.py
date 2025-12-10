import numpy as np
import scipy.io as sio
from scipy.signal import iirnotch, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from scipy.stats import skew, kurtosis
from collections import Counter


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

def window_signal(eeg_segment, window_size_sec=5, overlap=0.0):
    """
    Split EEG segment into non-overlapping windows.
    Parameters:
        eeg_segment: 1D array
        window_size_sec: window size in seconds
        overlap: fraction overlap (0.0 = no overlap)
    Returns:
        windows: array of shape (num_windows, window_samples)
    """
    window_samples = int(fs * window_size_sec)
    step = int(window_samples * (1 - overlap))
    windows = []
    
    for start in range(0, len(eeg_segment) - window_samples + 1, step):
        win = eeg_segment[start:start + window_samples]
        windows.append(win)
    
    return np.array(windows)

# Feature Extraction

def feature_1():
    pass

def feature_2(eeg_segment):
    """Compute mean, variance, skewness, kurtosis of raw EEG window"""
    return [
        np.mean(eeg_segment),
        np.var(eeg_segment),
        skew(eeg_segment),
        kurtosis(eeg_segment)
    ]

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


def accuracy(y_true, y_pred):
    """
    Compute classification accuracy.
    y_true: true labels
    y_pred: predicted labels
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(y_true == y_pred) / len(y_true)


def train_test_splitting(X, y, test_size=0.2, random_seed=42):
    np.random.seed(random_seed)
    
    classes = np.unique(y)
    train_indices = []
    test_indices = []
    
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        np.random.shuffle(cls_idx)
        split = int(len(cls_idx) * (1 - test_size))
        train_indices.extend(cls_idx[:split])
        test_indices.extend(cls_idx[split:])
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def classify_80_20(data, labels, fs, window_size_sec=5):
    """
    Perform 80-20 train/test split and classify using KNN.
    Compare the five representations
    """
    X_raw = []
    X_raw_stats = []
    X_derivative_signal = []
    X_deriv = []
    X_band = []
    y_all = []

    for i, segment in enumerate(data):
        filtered = process_50Hz(segment, fs)
        windows = window_signal(filtered, window_size_sec, overlap=0.0)

        for win in windows:
            # 1) Raw EEG samples
            X_raw.append(win)
            # 2) Raw EEG statistical features
            X_raw_stats.append(feature_2(win))
            # 3) First-order derivative (temporal difference)
            X_derivative_signal.append(feature_3(win))
            # 4) Derivative statistics
            X_deriv.append(feature_4(win))
            # 5) Frequency-band features
            X_band.append(feature_5(win, fs))
            # Label for this window
            y_all.append(labels[i])

    # Convert to arrays
    X_raw = np.array(X_raw)
    X_raw_stats = np.array(X_raw_stats)
    X_derivative_signal = np.array(X_derivative_signal)
    X_deriv = np.array(X_deriv)
    X_band = np.array(X_band)
    y_all = np.array(y_all)

    representations = {
        "Raw EEG": X_raw,
        "Raw EEG Stats": X_raw_stats,
        "Derivative signal": X_derivative_signal,
        "Derivative stats": X_deriv,
        "Frequency-band features": X_band
    }

    for name, X in representations.items():
        print(f"\nClassification using {name}:")
        X_train, X_test, y_train, y_test = train_test_splitting(X, y_all, test_size=0.2)

        acc_list = []
        for k in range(1, 11):
            y_pred = classify_classify_KNN(X_train, y_train, X_test, k)
            acc = accuracy(y_test, y_pred)
            acc_list.append(acc)
            print(f"K={k}, Accuracy={acc:.3f}")

        # Plot Accuracy vs K
        plt.figure()
        plt.plot(range(1, 11), acc_list, marker='o')
        plt.title(f"KNN Accuracy vs K ({name})")
        plt.xlabel("K")
        plt.ylabel("Accuracy")
        plt.xticks(range(1, 11))
        plt.grid(True)
        plt.show()

        # Print best K and accuracy
        best_k = np.argmax(acc_list) + 1  # +1 because K=1..10
        best_acc = acc_list[best_k-1]
        print(f"Best K for {name}: {best_k} with Accuracy = {best_acc:.3f}")


def classify_KNN(X_train, y_train, X_test, K):
    y_pred = []
    for x in X_test:
        # Compute Euclidean distances to all training samples
        distances = np.linalg.norm(X_train - x, axis=1)
        # Get indices of K nearest neighbors
        nn_idx = np.argsort(distances)[:K]
        # Get labels of nearest neighbors
        nn_labels = y_train[nn_idx]
        # Majority vote
        most_common = Counter(nn_labels).most_common(1)[0][0]
        y_pred.append(most_common)
    return np.array(y_pred)


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
    # First EEG segment 
    first_segment_raw = data[0, :]
    print("\nData amplitude ranges per class:")
    for cls in [0, 1, 2]:
        class_data = data[labels == cls]
        print(f"Class {cls}: min={class_data.min():.2f}, max={class_data.max():.2f}, std={class_data.std():.2f}")
   
    # Apply 50 Hz notch filter 
    first_segment_filtered = process_50Hz(first_segment_raw, fs)


    # Feature 2: Time domain stats
    feat2 = feature_2(first_segment_filtered)

    # Feature 3: First-order difference 
    feat3 = feature_3(first_segment_filtered)

    #Feature 4: Stats from derivative 
    feat4 = feature_4(first_segment_filtered)

    # Feature 5: Frequency band features
    feat5 = feature_5(first_segment_filtered, fs)

   

    #print("Feature 1 (raw) length:", len(first_segment_raw))
    

    #print("Feature 2 (raw) stats length:", len(feat2))
    #print("Feature 3 (Derivative) length:", len(feat3))
    #print("Feature 4 (Derivative stats):", feat4)
    #print("Feature 5 (Frequency band features) length:", len(feat5))

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
    print("Classification for window size = 5\n")
    classify_80_20(data, labels, fs, window_size_sec=5)
    print("Classification for window size = 10\n")
    classify_80_20(data, labels, fs, window_size_sec=10)
    print("Classification for window size = 15\n")
    classify_80_20(data, labels, fs, window_size_sec=15)
    print("Classification for window size = 20\n")
    classify_80_20(data, labels, fs, window_size_sec=20)

if __name__ == "__main__":
    main()
