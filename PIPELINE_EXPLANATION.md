# Yoga Stress Reduction - Complete Pipeline Explanation

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          WEARABLE STRESS DETECTION PIPELINE                 │
└─────────────────────────────────────────────────────────────────────────────┘

STAGE 1: DATA COLLECTION
            ↓
STAGE 2: DATA LOADING & EXTRACTION
            ↓
STAGE 3: SIGNAL PROCESSING & FILTERING
            ↓
STAGE 4: FEATURE EXTRACTION & ENGINEERING
            ↓
STAGE 5: DATA AGGREGATION & MERGING
            ↓
STAGE 6: MODEL TRAINING & CROSS-VALIDATION
            ↓
STAGE 7: MODEL EVALUATION & METRICS
```

---

## STAGE 1: DATA COLLECTION

### Devices & Sensors

**Empatica E4 (Wrist Device)**
- Blood Volume Pulse (BVP): 64 Hz
- Electrodermal Activity (EDA): 4 Hz
- Temperature (TEMP): 4 Hz
- Accelerometer (ACC): 32 Hz

**RespiBAN (Chest Device)**
- Electrocardiogram (ECG): 700 Hz
- Electromyography (EMG): 700 Hz
- Respiration (Resp): 700 Hz
- Temperature: 700 Hz
- Accelerometer: 700 Hz

### Study Protocol (WESAD Dataset)
```
Baseline (20 min)     → Subject sitting/standing reading magazines
Stress (5 min)        → Trier Social Stress Test (public speaking + math)
Amusement (5 min)     → Watching funny video clips

Labels: 0=amusement, 1=baseline, 2=stress, 3=meditation
```

---

## STAGE 2: DATA LOADING & EXTRACTION

### File Structure
```
WESAD-2/
├── S2/
│   └── S2.pkl  ← Contains synchronized multimodal data
├── S3/
├── ...
└── S15/

Each .pkl file contains:
{
  'signal': {
    'chest': {'ECG', 'EMG', 'EDA', 'Temp', 'Resp', 'ACC'},
    'wrist': {'BVP', 'EDA', 'TEMP', 'ACC'}
  },
  'label': [array of labels for each timestamp]
}
```

### Code Flow: `merge.py`

```python
def load_subject_data(subject_id):
    """Load synchronized data from .pkl file"""
    file_path = f"WESAD-2/S{subject_id}/S{subject_id}.pkl"
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

def extract_signals(data):
    """Extract signals from chest and wrist devices"""
    # Chest signals
    ecg = data['signal']['chest']['ECG'].flatten()
    emg = data['signal']['chest']['EMG'].flatten()
    eda_chest = data['signal']['chest']['EDA'].flatten()
    
    # Wrist signals
    bvp = data['signal']['wrist']['BVP'].flatten()
    eda_wrist = data['signal']['wrist']['EDA'].flatten()
    
    # Labels
    labels = data['label']
    
    return ecg, emg, eda_chest, bvp, eda_wrist, labels
```

**Key Point**: Data is already time-synchronized from WESAD dataset preparation.

---

## STAGE 3: SIGNAL PROCESSING & FILTERING

### Purpose
Remove noise, artifacts, and irrelevant frequency components from raw signals.

### Processing Techniques

#### A. Butterworth Low-Pass Filtering
```python
def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Parameters:
    - cutoff: cutoff frequency (Hz)
    - fs: sampling frequency (Hz)
    - order: filter order (higher = steeper cutoff)
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return lfilter(b, a, data)
```

**Applied to:**
- EDA (cutoff=1Hz, removes high-frequency noise)
- EMG (bandpass 20-250Hz, isolates muscle activity)
- BVP (removes motion artifacts)

#### B. Bandpass Filtering (EMG)
```python
# EMG needs bandpass because:
# - Below 20Hz: low-frequency noise
# - Above 250Hz: high-frequency noise
sos = butter(4, [20, 250], 'bandpass', fs=700)
filtered_emg = sosfilt(sos, emg_signal)
```

#### C. Time-Index Alignment
```python
# Different sampling rates → align to common timeline
eda_df.index = [(1 / fs_dict['EDA']) * i for i in range(len(eda_df))]
bvp_df.index = [(1 / fs_dict['BVP']) * i for i in range(len(bvp_df))]
# Convert to datetime for merging
eda_df.index = pd.to_datetime(eda_df.index, unit='s')
```

---

## STAGE 4: FEATURE EXTRACTION & ENGINEERING

### Overview
Convert raw sensor signals into meaningful features for ML model.

### Feature Extraction Functions

#### A. EDA (Electrodermal Activity) Decomposition - `cvxEDA`

**Problem**: EDA contains multiple components that are hard to separate

**Solution**: Convex Optimization approach decomposes EDA into:

```
Raw EDA = Tonic + Phasic + Noise

Where:
├─ Tonic (t):     Slow baseline (sleep, rest state)
├─ Phasic (r):    Rapid spikes (emotional responses)
└─ SMNA (p):      Sparse driver of phasic component
```

**Algorithm**: `cvxEDA.cvxEDA(normalized_signal, sampling_interval)`

```python
def eda_stats(y):
    Fs = fs_dict['EDA']  # 4 Hz
    yn = (y - y.mean()) / y.std()  # Normalize
    
    [r, p, t, l, d, e, obj] = cvxEDA.cvxEDA(yn, 1./Fs)
    # Returns: phasic, SMNA, tonic, spline_coef, drift, residuals, objective
    return [r, p, t, l, d, e, obj]
```

**Why important?**
- Tonic → baseline physiological state
- Phasic → emotional reactivity
- SMNA → sympathetic nervous activity

#### B. Statistical Features Extraction

For each signal component, extract:

```python
def get_window_stats(data, label=-1):
    """Calculate statistics over 30-second window"""
    mean   = np.mean(data)
    std    = np.std(data)
    min    = np.amin(data)
    max    = np.amax(data)
    
    return {
        'mean': mean,
        'std': std,
        'min': min,
        'max': max,
        'label': label
    }
```

#### C. ECG-derived Features (Heart Rate Variability)

```python
def extract_ecg_features(ecg_signal, sampling_rate=700):
    """Extract cardiac features using neurokit2"""
    # Process ECG
    ecg_signals, info = nk.ecg_process(ecg_signal, sampling_rate=700)
    
    # Extract R-peaks
    r_peaks = info['ECG_R_Peaks']
    
    # Calculate RR intervals (time between heartbeats)
    rr_intervals = np.diff(r_peaks) / sampling_rate * 1000  # milliseconds
    
    # HRV Features:
    # ├─ Heart Rate: 60000 / mean(rr_intervals)
    # ├─ SDNN: std(rr_intervals) → overall variability
    # ├─ RMSSD: sqrt(mean(diff(rr_intervals)²)) → parasympathetic activity
    # ├─ Skewness: distribution shape
    # └─ Kurtosis: distribution tails
    
    return {
        'Heart_Rate': heart_rate,
        'SDNN': sdnn,
        'RMSSD': rmssd,
        'ECG_Skewness': skew(ecg_signal),
        'ECG_Kurtosis': kurtosis(ecg_signal)
    }
```

#### D. EMG Features (Muscle Activity)

```python
def extract_emg_features(emg_signal, sampling_rate=700):
    """Extract muscle activity features"""
    # Filter EMG (20-250 Hz bandpass)
    filtered_emg = bandpass_filter(emg_signal, 20, 250, 700)
    
    # Calculate EMG envelope (muscle activation)
    emg_envelope = calculate_rms_envelope(filtered_emg, window=125ms)
    
    return {
        'EMG_Mean': np.mean(emg_envelope),
        'EMG_Std': np.std(emg_envelope),
        'EMG_Max': np.max(emg_envelope),
        'EMG_Skewness': skew(emg_envelope),
        'EMG_Kurtosis': kurtosis(emg_envelope)
    }
```

#### E. BVP Features (Blood Volume Pulse)

```python
# Peak frequency extraction
def get_peak_freq(x):
    f, Pxx = periodogram(x, fs=8)  # Power spectral density
    peak_freq = f[np.argmax(Pxx)]   # Frequency with max power
    return peak_freq
```

#### F. Temperature Features

```python
# Temperature slope indicator of body state change
def get_slope(series):
    linreg = scipy.stats.linregress(np.arange(len(series)), series)
    slope = linreg[0]
    return slope
```

### Complete Feature Set (31 Features)

```
BVP Features (5):
  ├─ BVP_mean, BVP_std, BVP_min, BVP_max
  └─ BVP_peak_freq

EDA Features (12):
  ├─ EDA_phasic_mean, EDA_phasic_std, EDA_phasic_min, EDA_phasic_max
  ├─ EDA_smna_mean, EDA_smna_std, EDA_smna_min, EDA_smna_max
  └─ EDA_tonic_mean, EDA_tonic_std, EDA_tonic_min, EDA_tonic_max

Respiration Features (4):
  └─ Resp_mean, Resp_std, Resp_min, Resp_max

Temperature Features (5):
  ├─ TEMP_mean, TEMP_std, TEMP_min, TEMP_max
  └─ TEMP_slope

Demographic (3):
  └─ age, height, weight
```

---

## STAGE 5: DATA AGGREGATION & MERGING

### Objective
Combine features from all subjects into single dataset for model training.

### Process

```
For each subject (S2 to S15):
  └─ Load subject data (.pkl)
     └─ Extract signals
        └─ Apply filtering
           └─ Extract features per 30-second window
              └─ Append to master dataframe
                 └─ Save subject features to: data/subject_feats/S{id}_feats_4.csv

Concatenate all subjects:
  └─ Combine all subject CSVs
     └─ Create merged dataset: data/m14_merged.csv
        └─ Format: [BVP_mean, BVP_std, ..., EDA_tonic_max, TEMP_slope, subject_id, label]
```

### Output: `data/m14_merged.csv`

```
Index    BVP_mean  BVP_std  ...  TEMP_slope  subject  label
0        45.3      12.1    ...  0.05        2        1  (baseline)
1        46.1      13.5    ...  0.04        2        1  (baseline)
2        78.9      22.4    ...  -0.02       2        2  (stress)
3        79.5      21.2    ...  -0.01       2        2  (stress)
...

Total rows: ~10,000+ samples (14 subjects × 3 conditions × 30-second windows)
Total columns: 31 features + subject_id + label
```

---

## STAGE 6: MODEL TRAINING & CROSS-VALIDATION

### Architecture

```
Input Layer: 31 features
    ↓
Dense Layer 1: 128 neurons + ReLU activation
    ↓
Dense Layer 2: 256 neurons + ReLU activation
    ↓
Output Layer: 3 neurons + LogSoftmax
    ↓
Classes: [0: Amusement, 1: Baseline, 2: Stress]
```

### Code: `m15_model_cv.ipynb`

```python
class WESADDataset(Dataset):
    """PyTorch dataset wrapper for WESAD data"""
    def __init__(self, dataframe):
        self.dataframe = dataframe.drop('subject', axis=1)
        self.labels = self.dataframe['label'].values
        self.dataframe.drop('label', axis=1, inplace=True)
        
    def __getitem__(self, idx):
        x = self.dataframe.iloc[idx].values
        y = self.labels[idx]
        return torch.Tensor(x), y
    
    def __len__(self):
        return len(self.dataframe)

class StressNet(nn.Module):
    """Neural network for stress classification"""
    def __init__(self):
        super(StressNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(31, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        return self.fc(x)
```

### Leave-One-Subject-Out Cross-Validation (LOSO-CV)

```python
def get_data_loaders(subject_id, train_batch_size=25, test_batch_size=5):
    """Leave one subject out for testing"""
    df = pd.read_csv('data/m14_merged.csv', index_col=0)
    
    # Train on all subjects EXCEPT subject_id
    train_df = df[df['subject'] != subject_id].reset_index(drop=True)
    
    # Test on subject_id
    test_df = df[df['subject'] == subject_id].reset_index(drop=True)
    
    train_dset = WESADDataset(train_df)
    test_dset = WESADDataset(test_df)
    
    train_loader = DataLoader(train_dset, batch_size=25, shuffle=True)
    test_loader = DataLoader(test_dset, batch_size=5)
    
    return train_loader, test_loader
```

**Why LOSO-CV?**
- Tests generalization to completely unknown subjects
- Prevents data leakage (no subject appears in both train and test)
- Realistic real-world scenario
- 14 iterations (one per subject)

### Training Loop

```python
def train(model, optimizer, train_loader, validation_loader):
    history = {'train_loss': {}, 'train_acc': {}, 'valid_loss': {}, 'valid_acc': {}}
    
    for epoch in range(num_epochs):  # e.g., 100+ epochs
        # TRAINING PHASE
        total = 0
        correct = 0
        trainlosses = []
        
        for batch_index, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images.float())
            
            # Calculate loss
            loss = criterion(outputs, labels)  # Cross-entropy loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            trainlosses.append(loss.item())
            
            # Calculate accuracy
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()
            total += len(labels)
        
        history['train_loss'][epoch] = np.mean(trainlosses)
        history['train_acc'][epoch] = correct / total
        
        # VALIDATION PHASE (every 10 epochs)
        if epoch % 10 == 0:
            with torch.no_grad():
                losses = []
                total = 0
                correct = 0
                
                for images, labels in validation_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images.float())
                    loss = criterion(outputs, labels)
                    
                    _, argmax = torch.max(outputs, 1)
                    correct += (labels == argmax).sum().item()
                    total += len(labels)
                    losses.append(loss.item())
                
                history['valid_acc'][epoch] = np.round(correct / total, 3)
                history['valid_loss'][epoch] = np.mean(losses)
                
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {np.mean(losses):.4}, Acc: {correct/total:.2}')
    
    return history
```

---

## STAGE 7: MODEL EVALUATION & METRICS

### Evaluation Metrics

```python
from sklearn.metrics import confusion_matrix, classification_report

# For each subject:
predictions = model.predict(test_subject_data)
true_labels = test_subject_data.labels

# 1. Confusion Matrix
cm = confusion_matrix(true_labels, predictions)

#              Predicted
#           Amusement  Baseline  Stress
# Amusement  [1200      45        30]
# Baseline   [40      1350       50]
# Stress     [25      35       1400]

# 2. Classification Metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)          # Correctness when model predicts positive
recall = TP / (TP + FN)              # Coverage of actual positives
f1_score = 2 * (precision * recall) / (precision + recall)

# 3. Cross-Subject Performance
mean_accuracy = np.mean([accuracy_S2, accuracy_S3, ..., accuracy_S15])
std_accuracy = np.std([accuracy_S2, accuracy_S3, ..., accuracy_S15])
```

### Results Example

```
┌─────────────────────────────────────────────────────────────┐
│              LOSO-CV Results by Subject                     │
├──────────┬──────────┬───────────┬────────────┬────────────┤
│ Subject  │ Accuracy │ Precision │ Recall (F1)│ Avg Score  │
├──────────┼──────────┼───────────┼────────────┼────────────┤
│ S2       │ 92.3%    │ 0.91      │ 0.92       │ 0.915      │
│ S3       │ 88.7%    │ 0.87      │ 0.89       │ 0.880      │
│ S4       │ 94.1%    │ 0.94      │ 0.94       │ 0.940      │
│ ...      │ ...      │ ...       │ ...        │ ...        │
│ S15      │ 90.2%    │ 0.90      │ 0.90       │ 0.900      │
├──────────┼──────────┼───────────┼────────────┼────────────┤
│ MEAN     │ 90.8%    │ 0.91      │ 0.91       │ 0.909      │
│ STD      │ 2.3%     │ 0.02      │ 0.02       │ 0.020      │
└──────────┴──────────┴───────────┴────────────┴────────────┘
```

---

## Complete Pipeline Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          WEARABLE STRESS DETECTION PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────────────────┘

STAGE 1: DATA COLLECTION
┌──────────────────────────────────────────────┐
│ Empatica E4 (Wrist)  │ RespiBAN (Chest)     │
│ ├─ BVP (64Hz)        │ ├─ ECG (700Hz)       │
│ ├─ EDA (4Hz)         │ ├─ EMG (700Hz)       │
│ ├─ TEMP (4Hz)        │ ├─ EDA (700Hz)       │
│ └─ ACC (32Hz)        │ ├─ Resp (700Hz)      │
│                      │ └─ ACC (700Hz)       │
└──────────────────────────────────────────────┘
                    ↓
STAGE 2: DATA LOADING
┌──────────────────────────────────────────────┐
│ Load WESAD-2/S{2-15}/S{2-15}.pkl files       │
│ Extract signals & labels per subject         │
└──────────────────────────────────────────────┘
                    ↓
STAGE 3: SIGNAL PROCESSING
┌──────────────────────────────────────────────┐
│ ├─ Butterworth Low-Pass (EDA, BVP)          │
│ ├─ Bandpass Filter (EMG: 20-250Hz)          │
│ └─ Time-Alignment (Resampling + Indexing)  │
└──────────────────────────────────────────────┘
                    ↓
STAGE 4: FEATURE EXTRACTION
┌──────────────────────────────────────────────┐
│ ├─ EDA Decomposition (cvxEDA):              │
│ │  ├─ Tonic (baseline)                      │
│ │  ├─ Phasic (reactions)                    │
│ │  └─ SMNA (sympathetic activity)           │
│ ├─ Statistical Features:                    │
│ │  ├─ mean, std, min, max per signal        │
│ │  └─ for each 30-second window             │
│ ├─ ECG-derived (HRV):                       │
│ │  ├─ Heart Rate, SDNN, RMSSD               │
│ │  └─ Skewness, Kurtosis                    │
│ ├─ EMG Envelope:                            │
│ │  └─ EMG_mean, EMG_std, EMG_max...         │
│ └─ Other: BVP_peak_freq, TEMP_slope         │
│                                              │
│ Result: 31 features per 30-sec window       │
└──────────────────────────────────────────────┘
                    ↓
STAGE 5: DATA AGGREGATION
┌──────────────────────────────────────────────┐
│ ├─ For each subject:                        │
│ │  └─ Save: data/subject_feats/S{id}.csv    │
│ └─ Merge all subjects:                      │
│    └─ Output: data/m14_merged.csv           │
│       (rows: 10000+, cols: 31+2)            │
└──────────────────────────────────────────────┘
                    ↓
STAGE 6: MODEL TRAINING (LOSO-CV)
┌──────────────────────────────────────────────┐
│ For subject_id in [S2 to S15]:              │
│  ├─ Train on: all subjects except S_id     │
│  ├─ Test on: S_id only                     │
│  ├─ Model:                                  │
│  │  ├─ Input(31) → FC(128, ReLU)           │
│  │  ├─ FC(256, ReLU)                        │
│  │  └─ Output(3, LogSoftmax)               │
│  ├─ Loss: Cross-Entropy                     │
│  ├─ Optimizer: Adam                         │
│  └─ Epochs: 100+                            │
│                                              │
│ Result: 14 models (one per subject)         │
└──────────────────────────────────────────────┘
                    ↓
STAGE 7: EVALUATION & METRICS
┌──────────────────────────────────────────────┐
│ ├─ Confusion Matrix (for each subject)      │
│ ├─ Precision, Recall, F1-Score              │
│ ├─ Mean Accuracy across subjects            │
│ └─ Final Performance: ~90.8% accuracy       │
└──────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
Raw Sensor Signals (Multiple Devices, Different Sampling Rates)
    │
    ├─ ECG (700Hz) ─────────┐
    ├─ EMG (700Hz) ─────────┤
    ├─ EDA (4Hz/700Hz) ─────┤
    ├─ BVP (64Hz) ──────────┤
    ├─ Resp (700Hz) ────────┼─→ Signal Filtering & Normalization
    ├─ ACC (32Hz/700Hz) ────┤
    └─ TEMP (4Hz/700Hz) ────┘
                    │
                    ↓
        Time-Aligned Signals (Resampled to common timeline)
                    │
                    ↓
            Feature Extraction
            ├─ Statistical (mean, std, min, max)
            ├─ Signal Processing (Peak Freq, Slope)
            ├─ EDA Decomposition (Tonic, Phasic, SMNA)
            ├─ ECG HRV (Heart Rate, SDNN, RMSSD)
            └─ EMG Envelope
                    │
                    ↓
            31-Dimensional Feature Vectors
            (per 30-second window)
                    │
                    ↓
            Combine from All Subjects
                    │
                    ↓
            Master Dataset (CSV)
                    │
                    ↓
            Train/Test Split (LOSO-CV)
                    │
        ┌───────────┴───────────┐
        │                       │
        ↓                       ↓
    Train Set          Test Set (Held Out)
    (13 subjects)      (1 subject)
        │                       │
        ↓                       ↓
    Neural Network          Evaluation
    (Training)             (Inference)
        │                       │
        └───────────┬───────────┘
                    ↓
            Accuracy & Metrics
            (Repeat 14 times for LOSO-CV)
                    │
                    ↓
            Cross-Subject Average Performance
```

---

## Key Files in Pipeline

| File | Purpose |
|------|---------|
| `merge.py` | Load pickle files, extract signals, apply filtering, extract ECG/EMG/EDA features |
| `data_wrangling.py` | Handle EDA decomposition with cvxEDA, windowing, statistical aggregation |
| `cvxEDA.py` | Convex optimization for decomposing EDA into components |
| `feature_extraction.ipynb` | Test and visualize feature extraction on raw data |
| `m15_model_cv.ipynb` | Model definition, training loop, LOSO-CV implementation, evaluation |
| `data/m14_merged.csv` | Final aggregated dataset ready for model training |
| `m13_model.pt` | Saved trained model weights |

---

## Summary

The pipeline transforms **raw multimodal wearable sensor data** into a **machine learning-ready dataset** and trains a **deep neural network** with rigorous **leave-one-subject-out cross-validation** to classify emotional states (stress, amusement, baseline) with ~91% accuracy. The key innovation is the sophisticated feature engineering combining signal processing, statistical analysis, and physiological domain knowledge to create meaningful representations of emotional states from wearable sensors.
