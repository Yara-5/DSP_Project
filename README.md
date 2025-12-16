# EEG Signal Classification for Epilepsy Detection using K-Nearest Neighbors (KNN)

## Introduction

This project implements an automated epilepsy detection system that classifies EEG signals into three brain states:

- **Rest**
- **Active**
- **Seizure**

The dataset consists of **500 EEG segments**, each containing **4096 samples** recorded at **173 Hz sampling rate** (~23.6 seconds per segment).  

The approach involves:  

1. **Preprocessing**: Apply a 50 Hz notch filter to remove electrical interference.  
2. **Feature Extraction**: Compute five different feature representations:
   - Raw time-domain signals
   - Statistical features (mean, variance, skewness, kurtosis)
   - Differentiated (derivative) signals
   - Derivative statistics
   - Frequency-band features (delta, theta, alpha, beta, gamma)  
3. **Classification**: Use **K-Nearest Neighbors (KNN)** with K values ranging from 1 to 10.  

The dataset is split **80% training / 20% testing**, and experiments are conducted using window lengths of **5, 10, 15, and 20 seconds** to evaluate their impact on classification performance. This automated approach shows promise for **real-time seizure detection in clinical settings**.

---

## Project Pipeline

The project implements a complete **EEG classification pipeline**:

1. **Preprocessing**
   - Each segment is filtered with a **50 Hz notch filter** to remove mains power interference while preserving EEG content.
   - Segments are divided into windows of length 5, 10, 15, and 20 seconds. Each window is treated as an individual sample for feature extraction and classification.

2. **Feature Extraction**
   For every window, five feature representations are computed:
   
   - **Time-Domain Features**:  
     - Raw filtered samples  
     - Statistical summaries: mean, variance, skewness, kurtosis
   - **Derivative Features**:  
     - First-order difference of the signal  
     - Statistical summaries of the derivative
   - **Frequency-Domain Features**:  
     - Decomposition into standard EEG bands: delta, theta, alpha, beta, gamma  
     - Four statistics per band (mean, variance, skewness, kurtosis)  

   These features allow comparison between raw signals, time-domain summaries, derivative-based features, and frequency-based features.

3. **Classification**
   - Label windows according to their parent segment’s class:
     - `0 = Rest`
     - `1 = Active`
     - `2 = Seizure`
   - Split features into **80% training / 20% testing** sets using **stratified sampling** to preserve class proportions.
   - Apply **K-Nearest Neighbors (KNN)** classifier with **K = 1–10**.  
   - For each K, predict labels of test windows by **majority voting** among the K closest training windows.  
   - Compute **classification accuracy** and plot **accuracy vs K** curves.  

This framework allows systematic comparison of:

- Raw vs. statistical features  
- Derivative-based vs. raw features  
- Time-domain vs. frequency-band features  
- Effect of window length on classification performance  

---

## Results

The pipeline provides insights into:

- Which feature representation yields the **highest accuracy**  
- Optimal **window length** for classification  
- Best **K value** for KNN for each feature type  

These results can inform **real-time EEG seizure detection applications**.

---

## Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `scipy`
  - `scikit-learn`
  - `matplotlib`
  - `pandas`

---

## Usage

1. Clone the repository:  
   ```bash
   git clone https://github.com/YourUsername/EEG-Classification.git
2. Navigate to the project folder:
   ```bash
   cd EEG-Classification
3. Run the main script:
   ```bash
   python eeg_classification.py
   
## License
This project is licensed under the MIT License.
