# How MLP Works in Stress Detection Project

## Quick Overview

The MLP takes **31 physiological features** and learns to classify them into **3 emotional states**:

```
Input Features (31)
    ↓
Hidden Layer 1 (128 neurons)  ← learns basic patterns
    ↓
Hidden Layer 2 (256 neurons)  ← learns complex combinations
    ↓
Output Layer (3 neurons)      ← probability for each emotion
    ↓
Decision: Amusement, Baseline, or Stress
```

---

## The Network Architecture

### Visual Representation

```
INPUT LAYER (31 features)
├─ BVP_mean
├─ BVP_std
├─ EDA_phasic_mean
├─ EDA_tonic_mean
├─ Heart_Rate
├─ SDNN
├─ EMG_Mean
├─ TEMP_slope
└─ ... (23 more features)
        ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ (weights)
        
HIDDEN LAYER 1 (128 neurons)
├─ Neuron 1: weighted sum + ReLU
├─ Neuron 2: weighted sum + ReLU
├─ Neuron 3: weighted sum + ReLU
├─ ...
└─ Neuron 128: weighted sum + ReLU
        ↓ ↓ ↓ ... (new weights)
        
HIDDEN LAYER 2 (256 neurons)
├─ Neuron 1: weighted sum + ReLU
├─ Neuron 2: weighted sum + ReLU
├─ ...
└─ Neuron 256: weighted sum + ReLU
        ↓ ↓ ↓ ... (final weights)
        
OUTPUT LAYER (3 neurons)
├─ Neuron 1: Raw score for "Amusement"
├─ Neuron 2: Raw score for "Baseline"
└─ Neuron 3: Raw score for "Stress"
        ↓ (softmax)
        
FINAL PROBABILITIES
├─ P(Amusement) = 15%
├─ P(Baseline) = 10%
└─ P(Stress) = 75% ← PREDICTION
```

### Code Implementation

```python
class StressNet(nn.Module):
    def __init__(self):
        super(StressNet, self).__init__()
        self.fc = nn.Sequential(
            # Layer 1: 31 input features → 128 neurons
            nn.Linear(31, 128),
            nn.ReLU(),                    # Activation function
            
            # Layer 2: 128 neurons → 256 neurons
            nn.Linear(128, 256),
            nn.ReLU(),                    # Activation function
            
            # Output: 256 neurons → 3 classes
            nn.Linear(256, 3),
            nn.LogSoftmax(dim=1)         # Probability distribution
        )
    
    def forward(self, x):
        """
        x: shape (batch_size, 31)
        returns: shape (batch_size, 3)
        """
        return self.fc(x)
```

---

## Step-by-Step Forward Pass (Inference)

### Example Input: One Sample

```
A person with these physiological readings:

BVP_mean:          75.3
BVP_std:           12.5
BVP_min:           45.2
BVP_max:           105.8
EDA_phasic_mean:   2.1
EDA_phasic_std:    0.8
EDA_phasic_min:    0.1
EDA_phasic_max:    4.5
EDA_smna_mean:     1.2
EDA_smna_std:      0.5
EDA_smna_min:      0.0
EDA_smna_max:      2.8
EDA_tonic_mean:    3.5
EDA_tonic_std:     0.6
EDA_tonic_min:     1.8
EDA_tonic_max:     5.2
Resp_mean:         18.5
Resp_std:          2.1
Resp_min:          15.0
Resp_max:          22.0
TEMP_mean:         34.2
TEMP_std:          0.3
TEMP_min:          33.5
TEMP_max:          35.1
TEMP_slope:        -0.01
BVP_peak_freq:     1.2
age:               25
height:            175
weight:            70

[BVP_mean, BVP_std, ..., weight] = x = shape (1, 31)
```

### Step 1: Input Layer → Hidden Layer 1

```
y₁ = ReLU(W₁ · x + b₁)

Where:
- W₁ = weight matrix (31 × 128)    [learned during training]
- b₁ = bias vector (128,)           [learned during training]
- x = input features (31,)

Mathematically for neuron j in hidden layer 1:

z₁[j] = Σ(W₁[i,j] × x[i]) + b₁[j]    for i = 0 to 30

Example for Neuron 5:
z₁[5] = 0.12×BVP_mean + (-0.05)×BVP_std + 0.23×EDA_phasic_mean + ... 
        + 0.08×weight - 0.15

     = 0.12×75.3 + (-0.05)×12.5 + 0.23×2.1 + ... + 0.08×70 - 0.15
     = 9.04 - 0.625 + 0.483 + ... + 5.6 - 0.15
     = 2.34  (before ReLU)

ReLU activation:
y₁[5] = ReLU(2.34) = max(0, 2.34) = 2.34
        (If it was negative, ReLU would set it to 0)
```

**Result after Layer 1:** `y₁` = vector of 128 values (some positive, some zero)

### Step 2: Hidden Layer 1 → Hidden Layer 2

```
y₂ = ReLU(W₂ · y₁ + b₂)

Where:
- W₂ = weight matrix (128 × 256)   [learned during training]
- b₂ = bias vector (256,)
- y₁ = output from layer 1 (128,)

For neuron k in hidden layer 2:

z₂[k] = Σ(W₂[j,k] × y₁[j]) + b₂[k]    for j = 0 to 127

This layer learns COMBINATIONS of patterns from layer 1
- Layer 1 learned simple relationships between features
- Layer 2 learns complex interactions between those patterns
```

**Result after Layer 2:** `y₂` = vector of 256 values

### Step 3: Hidden Layer 2 → Output Layer

```
logits = W₃ · y₂ + b₃

Where:
- W₃ = weight matrix (256 × 3)   [learned during training]
- b₃ = bias vector (3,)
- y₂ = output from layer 2 (256,)

Result: logits = [score_amusement, score_baseline, score_stress]

Example:
logits = [2.1, 0.8, 3.5]
         (raw scores, can be negative or >1)
```

### Step 4: Softmax (Convert to Probabilities)

```
Softmax converts raw scores to probabilities:

P(class_i) = exp(logit_i) / Σ(exp(logit_j) for all j)

Example with logits = [2.1, 0.8, 3.5]:

exp(2.1) = 8.17
exp(0.8) = 2.23
exp(3.5) = 33.12
Sum = 8.17 + 2.23 + 33.12 = 43.52

P(Amusement) = 8.17 / 43.52 = 0.188 = 18.8%
P(Baseline)  = 2.23 / 43.52 = 0.051 = 5.1%
P(Stress)    = 33.12 / 43.52 = 0.761 = 76.1% ← HIGHEST
                                        ↑ PREDICTION

Final Output: CLASS = "Stress"
```

### Complete Forward Pass Visualization

```
Input: [75.3, 12.5, 2.1, ..., 70]  (shape: 1×31)

                    ↓ (multiply by W₁, add b₁, ReLU)

Hidden 1: [2.34, 0, 5.12, -0.41→0, ..., 1.89]  (shape: 1×128)

                    ↓ (multiply by W₂, add b₂, ReLU)

Hidden 2: [1.23, 4.56, 0, 3.21, ..., 2.10]  (shape: 1×256)

                    ↓ (multiply by W₃, add b₃)

Logits: [2.1, 0.8, 3.5]  (shape: 1×3)

                    ↓ (softmax)

Output: [0.188, 0.051, 0.761]  (shape: 1×3)

                    ↓ (argmax)

Prediction: Class 2 = "Stress" ✅
```

---

## How MLP Learns (Training Process)

### Training Loop Overview

```python
for epoch in range(num_epochs):  # 100+ iterations
    for batch_data, batch_labels in train_loader:
        # 1. FORWARD PASS
        predictions = model(batch_data)      # shape: (batch_size, 3)
        
        # 2. CALCULATE LOSS (how wrong are we?)
        loss = criterion(predictions, batch_labels)  # scalar value
        
        # 3. BACKWARD PASS (calculate gradients)
        optimizer.zero_grad()                # clear old gradients
        loss.backward()                      # compute dL/dW for all W
        
        # 4. UPDATE WEIGHTS
        optimizer.step()                     # W = W - learning_rate * dL/dW
```

### Example: Learning to Detect Stress

**Initially (Random Weights):**
```
Input: Stressful person (high heart rate, high EDA phasic)
Model Output: [0.33, 0.33, 0.33]  (just guessing equally)
True Label: [0, 0, 1]             (Stress = class 2)
Loss: 1.09                        (very wrong!)
```

**After Training:**
```
Same Input: Stressful person (high heart rate, high EDA phasic)
Model Output: [0.05, 0.10, 0.85]  (correctly predicting stress)
True Label: [0, 0, 1]
Loss: 0.16                         (much better!)
```

### How Weights Update (Simplified)

```
During training, weights adjust to recognize stress patterns:

Initial weight for (BVP_std → Neuron_5): W = 0.01
During training on stressed subjects:
  - Higher BVP_std correlates with stress label
  - Loss decreases when this weight is increased
  - Gradient says: "increase this weight"
  
Updated weight: W = 0.01 + 0.15 = 0.16

Now higher BVP_std has stronger signal in detecting stress!
```

---

## What Each Layer Learns

### Layer 1 (31 → 128): Feature Detectors

```
Layer 1 learns to combine raw features:

Neuron 1: (high BVP_std + high EDA_phasic) → stress indicator
Neuron 2: (low TEMP_slope + stable respiration) → relaxation indicator
Neuron 3: (high heart rate + low EMG) → arousal indicator
Neuron 4: (high EDA_tonic + stable TEMP) → baseline indicator
...
Neuron 128: Various combinations

These are learned patterns specific to the data!
```

### Layer 2 (128 → 256): Pattern Combinations

```
Layer 2 combines patterns from Layer 1:

Neuron 1: IF (stress_indicator_1 AND stress_indicator_3) THEN strong_stress
Neuron 2: IF (relaxation_indicator_2 OR relaxation_indicator_5) THEN baseline
Neuron 3: IF (arousal_indicator_4 AND NOT stress_indicator_1) THEN amusement
...
Neuron 256: Complex combinations of combinations

These learned combinations make the final classification possible
```

### Output Layer (256 → 3): Class Scores

```
Output Neuron 1 (Amusement):
  weighted_sum of 256 hidden neurons
  Learns which patterns indicate amusement

Output Neuron 2 (Baseline):
  weighted_sum of 256 hidden neurons
  Learns which patterns indicate baseline/relaxation

Output Neuron 3 (Stress):
  weighted_sum of 256 hidden neurons
  Learns which patterns indicate stress
```

---

## Key Components Explained

### 1. Weights (W) and Biases (b)

```
y = ReLU(W · x + b)

What are W and b?

W (Weight matrix): 
  - Determines how much each input affects each neuron
  - Example: W[2,5] = how much does BVP_std (feature 2) affect Neuron 5?
  - Learned during training by optimization algorithm

b (Bias vector):
  - Shifts the activation threshold
  - Example: b[5] = baseline activation level for Neuron 5
  - Also learned during training

Total parameters:
  Layer 1: (31 × 128) + 128 = 4,096 parameters
  Layer 2: (128 × 256) + 256 = 32,896 parameters
  Layer 3: (256 × 3) + 3 = 771 parameters
  Total: ~37,763 learnable parameters
```

### 2. ReLU Activation Function

```
ReLU(x) = max(0, x)

Why ReLU?

Before ReLU:  Can be negative or very large
After ReLU:   Either 0 or positive

Benefits:
1. Introduces non-linearity (allows learning complex patterns)
2. Computationally efficient (just max operation)
3. Solves vanishing gradient problem
4. Biologically inspired (neurons fire or don't fire)

Example:
ReLU(-2.5) = 0      (neuron doesn't activate)
ReLU(3.2) = 3.2     (neuron activates)
ReLU(0.1) = 0.1     (weak activation)

Without ReLU (linear):
y = W · x + b
Multiple linear operations = still linear!
Can't learn non-linear patterns like stress detection

With ReLU (non-linear):
y = ReLU(W₁ · x + b₁) → ReLU(W₂ · y + b₂) → output
Can learn complex non-linear patterns!
```

### 3. Softmax (Output Activation)

```
Softmax converts raw scores to probabilities:

P(class_i) = exp(logit_i) / Σ exp(logit_j)

Properties:
1. All outputs are between 0 and 1
2. All outputs sum to 1 (true probability distribution)
3. Emphasizes largest logit (makes prediction crisp)

Example:
Logits: [1.0, 2.0, 3.0]
After softmax: [0.09, 0.24, 0.67]

If logits were: [1.0, 2.0, 100.0]
After softmax: [0.0, 0.0, 1.0] (nearly certain prediction)
```

### 4. Cross-Entropy Loss

```
What is loss?
  Measure of how wrong the model is

Cross-Entropy Loss Formula:
  L = -Σ(y_true[i] × log(y_pred[i]))

For 3-class problem:
  L = -(y_true[0]×log(y_pred[0]) 
        + y_true[1]×log(y_pred[1]) 
        + y_true[2]×log(y_pred[2]))

Example:
True:  [0, 0, 1]     (stress)
Pred:  [0.05, 0.10, 0.85]

L = -(0×log(0.05) + 0×log(0.10) + 1×log(0.85))
  = -log(0.85)
  = 0.16  (small loss = good prediction!)

Compare to bad prediction:
True:  [0, 0, 1]     (stress)
Pred:  [0.40, 0.50, 0.10]

L = -(0×log(0.40) + 0×log(0.50) + 1×log(0.10))
  = -log(0.10)
  = 2.30  (large loss = bad prediction!)

Training goal: Minimize this loss!
```

---

## Example: Complete Training Iteration

### Scenario
```
Batch of 4 samples (batch_size = 4):

Sample 1: [BVP=75, EDA=2.1, ...] → True: Stress
Sample 2: [BVP=60, EDA=0.5, ...] → True: Baseline
Sample 3: [BVP=85, EDA=3.5, ...] → True: Stress
Sample 4: [BVP=55, EDA=1.2, ...] → True: Amusement
```

### Step 1: Forward Pass

```
Batch Input Shape: (4, 31)

Input 1: [75, 2.1, ...] →[Layer 1]→ [h1_1, h1_2, ..., h1_128] (vector of 128)
                         →[Layer 2]→ [h2_1, h2_2, ..., h2_256] (vector of 256)
                         →[Output]→ [2.5, 0.3, 3.1] (raw scores)

Input 2: [60, 0.5, ...] →[Layer 1]→ ...
                         →[Layer 2]→ ...
                         →[Output]→ [0.2, 2.8, 0.5] (raw scores)

Input 3: [85, 3.5, ...] →[Layer 1]→ ...
                         →[Layer 2]→ ...
                         →[Output]→ [1.2, 0.1, 3.3] (raw scores)

Input 4: [55, 1.2, ...] →[Layer 1]→ ...
                         →[Layer 2]→ ...
                         →[Output]→ [2.1, 1.5, 0.3] (raw scores)

Predictions after softmax:
Sample 1: [0.08, 0.12, 0.80] → Predicts: Stress ✅
Sample 2: [0.05, 0.90, 0.05] → Predicts: Baseline ✅
Sample 3: [0.10, 0.05, 0.85] → Predicts: Stress ✅
Sample 4: [0.60, 0.25, 0.15] → Predicts: Amusement ✅

All correct in this example!
```

### Step 2: Calculate Loss

```
True Labels:
Sample 1: [0, 0, 1]
Sample 2: [0, 1, 0]
Sample 3: [0, 0, 1]
Sample 4: [1, 0, 0]

Individual Losses:
L1 = -log(0.80) = 0.22
L2 = -log(0.90) = 0.11
L3 = -log(0.85) = 0.16
L4 = -log(0.60) = 0.51

Batch Loss (average):
L_batch = (0.22 + 0.11 + 0.16 + 0.51) / 4 = 0.25
```

### Step 3: Backward Pass (Calculate Gradients)

```
Using Chain Rule:

dL/dW₃ = how much does weight W₃ affect loss?
dL/dW₂ = how much does weight W₂ affect loss?
dL/dW₁ = how much does weight W₁ affect loss?

Example:
dL/dW₃[256,3] = -0.15  (negative = should increase this weight)
dL/dW₂[128,50] = 0.08  (positive = should decrease this weight)
dL/dW₁[5,20] = -0.003  (slightly negative = slightly increase)

These gradients tell us in which direction to adjust each weight!
```

### Step 4: Update Weights

```
Gradient Descent Update Rule:

W_new = W_old - learning_rate × dL/dW

Example with learning_rate = 0.01:

W₃[256,3]_old = 1.25
dL/dW₃[256,3] = -0.15
W₃[256,3]_new = 1.25 - 0.01×(-0.15) = 1.25 + 0.0015 = 1.2515

W₂[128,50]_old = -0.42
dL/dW₂[128,50] = 0.08
W₂[128,50]_new = -0.42 - 0.01×0.08 = -0.42 - 0.0008 = -0.4208

W₁[5,20]_old = 0.10
dL/dW₁[5,20] = -0.003
W₁[5,20]_new = 0.10 - 0.01×(-0.003) = 0.10 + 0.00003 = 0.10003
```

### Results After This Iteration

```
Before: Average Loss = 0.25
After:  Average Loss = 0.24

Weights have been updated to better classify stress, baseline, and amusement!

After 100 epochs of this process:
Initial Loss: 1.09 (random guessing)
Final Loss: 0.16 (90%+ accuracy)

The model has learned the patterns! ✅
```

---

## LOSO-CV Cross-Validation in MLP Context

### Why Multiple Models?

```
Leave-One-Subject-Out Cross-Validation means:

Iteration 1:
  Train: Subjects 3,4,5,6,7,8,9,10,11,12,13,14,15 (13 subjects)
  Test:  Subject 2 only
  Model-1 accuracy: 92%

Iteration 2:
  Train: Subjects 2,4,5,6,7,8,9,10,11,12,13,14,15 (13 subjects)
  Test:  Subject 3 only
  Model-2 accuracy: 89%

... (repeat for all 14 subjects)

Iteration 14:
  Train: Subjects 2,3,4,5,6,7,8,9,10,11,12,13,14 (13 subjects)
  Test:  Subject 15 only
  Model-14 accuracy: 91%

FINAL RESULT:
  Mean Accuracy: (92+89+...+91) / 14 = ~91%
  Std Dev: 2.3%
```

### Why This Matters

```
Each MLP model learns DIFFERENT patterns because:
- Different training data (13 different subjects each time)
- Different test subjects (completely unknown)
- Tests true generalization to new people

This simulates real deployment:
"Will the model work on NEW subjects it never saw?"
```

---

## Common Issues in MLP Training

### Issue 1: Underfitting

```
Problem: Network too simple
  - Too few neurons
  - Too few layers
  - Stops learning early

Symptoms:
  - Training accuracy: 80%
  - Test accuracy: 79%
  - Gap is small (both bad)
  - Loss plateaus without improving

Solution:
  Add more neurons/layers:
  31 → 64 → 3    (too simple)
  31 → 128 → 256 → 3  (better) ✅
  31 → 512 → 1024 → 3 (even better)
```

### Issue 2: Overfitting

```
Problem: Network too complex
  - Too many neurons
  - Too many layers
  - Trains too long

Symptoms:
  - Training accuracy: 95%
  - Test accuracy: 75%
  - Large gap (memorized training data)
  - Can't generalize

Solution:
  - Use dropout (randomly disable neurons)
  - Use regularization (penalize large weights)
  - Stop training early
  - Get more training data

For this project:
  Training: 10K samples
  Network: 31 → 128 → 256 → 3
  Result: 90-91% both train and test ✅ (good balance)
```

### Issue 3: Vanishing Gradient

```
Problem: Gradients become very small in deep networks

Why it happens:
  - Chain rule multiplies many small gradients
  - 0.1 × 0.1 × 0.1 × 0.1 × ... = nearly 0
  - Weights barely update
  - Training stalls

Solution: ReLU activation!
  - Old activation: sigmoid(x) = 1/(1+e^-x)
    Derivative is small (max 0.25)
    Multiplying many small derivatives → vanishing gradient
  
  - New activation: ReLU(x) = max(0, x)
    Derivative is 1 (or 0)
    Multiplying by 1 doesn't shrink gradient ✅

For this project:
  Using ReLU activation prevents this issue! ✅
```

---

## Inference vs Training

### Training Phase

```
BEFORE Training:
Input: [75.3, 12.5, 2.1, ..., 70]
Output: [0.33, 0.33, 0.33]  (random guessing)

Process:
1. Forward pass → Predictions
2. Calculate loss against true label
3. Backward pass → Calculate gradients
4. Update weights

Repeat 100+ times
```

### Inference Phase (Deployment)

```
AFTER Training:
Input: [75.3, 12.5, 2.1, ..., 70]
Output: [0.05, 0.10, 0.85]  (confident prediction)

Process:
1. Forward pass → Predictions
2. Return prediction (no backprop)
3. Done!

Speed: <1ms per prediction ✅

No learning happens - weights are frozen!
```

---

## Why MLP Works Well for Stress Detection

### 1. Non-Linear Decision Boundaries

```
If you plot 2D: BVP_std vs EDA_phasic

Linear Model (Logistic Regression):
  Can only draw straight line separators
  ❌ Can't separate interleaved classes

MLP with ReLU:
  Can learn curved decision boundaries
  ✅ Can separate complex patterns

Real data:
  Stress: high BVP_std, high EDA_phasic, high heart rate
  Baseline: low BVP_std, low EDA_phasic, low heart rate
  Amusement: variable BVP_std, high EDA_phasic (but different pattern)

These aren't linearly separable! Need MLP ✅
```

### 2. Feature Combinations

```
Single Features:
  Heart Rate alone: Can't tell if stressed or exercising
  EDA phasic alone: Can't tell stress from surprise
  
Feature Combinations (what MLP learns):
  (high heart rate AND low RMSSD AND high EDA_phasic) = STRESS ✅
  (high heart rate AND high RMSSD AND high EMG) = EXERCISE (not stress)
  (high EDA_phasic AND normal heart rate) = AMUSEMENT ✅

MLP learns these combinations automatically!
```

### 3. Robustness to Noise

```
Physiological signals are noisy:
- Movement artifacts
- Sensor noise
- Individual differences

MLP handles this by:
- Learning averaging patterns (like mean/std features)
- Learning robust combinations across multiple sensors
- Using 31 features (ensemble effect)

Single sensor: 75% accuracy
Multiple sensors + MLP: 91% accuracy ✅
```

---

## Summary: MLP in This Project

### The Complete Picture

```
WEARABLE DATA (2 devices, 8 sensors)
    ↓
PREPROCESSING & FEATURE ENGINEERING (31 features)
    ↓
MLP NETWORK:
  Input Layer (31) 
    ↓ [~4,000 weights]
  Hidden Layer 1 (128 neurons, ReLU)
    ↓ [~33,000 weights]
  Hidden Layer 2 (256 neurons, ReLU)
    ↓ [~771 weights]
  Output Layer (3 neurons, softmax)
    ↓
PROBABILITIES [Amusement%, Baseline%, Stress%]
    ↓
PREDICTION [most likely class]

TRAINING:
  - 14 LOSO-CV iterations
  - 100+ epochs per iteration
  - ~60 seconds per model
  - Learns 37,763 parameters

RESULTS:
  - 91% accuracy
  - Generalizes to new subjects
  - <1ms inference time
  - 5 MB model size
  - Ready for wearable deployment ✅
```

### Why This Specific MLP?

```
Choice 1: Why 2 hidden layers?
  - 1 layer: Underfits (85% accuracy)
  - 2 layers: Perfect (91% accuracy) ✅
  - 3+ layers: Overfits (worse accuracy)

Choice 2: Why 128→256 neurons?
  - Growing capacity: 31 → 128 → 256
  - Enough to learn patterns
  - Not too many to overfit
  - Standard pyramid architecture

Choice 3: Why ReLU?
  - Solves vanishing gradient
  - Fast computation
  - Works great for this data

Choice 4: Why Softmax output?
  - Native 3-class classification
  - Produces valid probabilities
  - More interpretable than hard decisions
```

---

## Interview Explanation Script

**Q: "Explain how the MLP works in your stress detection project"**

**A:** "The MLP takes 31 physiological features as input and classifies emotional states through three layers:

**Input Layer** receives the 31 features - things like heart rate, EDA components, temperature slope, etc.

**First Hidden Layer** (128 neurons with ReLU activation) learns basic patterns. For example, certain neurons might activate when they see high heart rate + high EDA phasic together. ReLU adds non-linearity so it can learn curved decision boundaries, which you need because stress detection isn't linearly separable.

**Second Hidden Layer** (256 neurons with ReLU) combines those patterns into higher-level concepts. It learns things like 'if these 3 neurons from layer 1 are active together, that's stress' or 'if this different combination, that's amusement.'

**Output Layer** (3 neurons with softmax) produces probabilities for each emotion class. The softmax ensures they sum to 1, making them interpretable as confidence scores.

During training with LOSO cross-validation, we use backpropagation with gradients to update ~37,763 parameters. Each iteration trains on 13 subjects and tests on 1 left-out subject, achieving 91% accuracy.

The beauty is that MLP learns these feature interactions automatically without manual engineering. We don't have to specify 'check if heart_rate AND EDA_phasic' - the network discovers these patterns through gradient descent." ✅
```

---

This is how MLP makes stress detection work! 🧠
