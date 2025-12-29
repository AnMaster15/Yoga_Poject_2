# -*- coding: utf-8 -*-
"""
Full-Workflow Script for Stacked Ensemble Classification with Explainability.

This script demonstrates a complete machine learning pipeline for a classification task,
intended for research purposes. The workflow includes:
1.  Data loading and preprocessing.
2.  Definition of four distinct deep learning architectures (DNN, Conv-LSTM, TDNN, Gated-CNN).
3.  Performance evaluation using 5-fold stratified cross-validation.
4.  Training a CatBoost meta-model on out-of-fold predictions (stacking).
5.  Training final models on the entire dataset.
6.  Generating and visualizing explanations for both the meta-model and base models
    using SHAP, LIME, and Grad-CAM to interpret model behavior.

To Run:
1. Place your dataset 'custom(1).csv' in the same directory.
2. Install the required libraries:
   pip install pandas numpy scikit-learn catboost tensorflow lime shap tf-keras-vis matplotlib
"""

# --- 1. IMPORTS & SETUP ---
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Models
from catboost import CatBoostClassifier
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, LSTM, Conv1D, GlobalAveragePooling1D, Dropout,
                                     BatchNormalization, Input, Bidirectional, LeakyReLU,
                                     concatenate, multiply)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.utils import to_categorical

# Explainability (XAI)
import shap
import lime
import lime.lime_tabular
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore

from matplotlib import cm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
tf.get_logger().setLevel('ERROR')

# --- 2. CONFIGURATION & HYPERPARAMETERS ---
# Data & Model Configuration
DATA_FILE = 'custom(1).csv'
TARGET_COLUMN = 'Label'
RANDOM_STATE = 42
N_SPLITS = 5

# Data Shape Configuration (based on feature engineering)
TIMESTEPS = 27
FEATURES_PER_TIMESTEP = 18 # This should be total_features / timesteps

# Training Hyperparameters
EPOCHS = 300
BATCH_SIZE = 64
PATIENCE = 40
LEARNING_RATE = 1e-3

# --- 3. DATA LOADING & PREPARATION ---
print("--- Step 1: Loading and Preparing Data ---")
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Data loaded successfully: {df.shape}")
except FileNotFoundError:
    print(f"Error: '{DATA_FILE}' not found. Creating a dummy dataset for demonstration.")
    num_samples = 5750
    num_features = TIMESTEPS * FEATURES_PER_TIMESTEP
    df = pd.DataFrame(np.random.rand(num_samples, num_features), columns=[f'feature_{i}' for i in range(num_features)])
    df[TARGET_COLUMN] = np.random.randint(0, 5, num_samples)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({i: f'class_{i}' for i in range(5)})

# Separate features and target
X = df.drop(columns=[TARGET_COLUMN]).values
X[np.isinf(X)] = np.nan # Handle potential infinite values

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df[TARGET_COLUMN])
N_CLASSES = len(le.classes_)
CLASS_NAMES = le.classes_

print(f"Dataset shape: {X.shape}, Number of classes: {N_CLASSES}\n")

# --- 4. MODEL ARCHITECTURES ---
# (Functions are identical to your original code)
def create_deep_dnn_model(input_shape, n_classes):
    inputs = Input(shape=input_shape)
    x = Dense(1024, activation=LeakyReLU())(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation=LeakyReLU())(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation=LeakyReLU())(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='Deep-DNN')

def create_deep_conv_lstm_model(input_shape, n_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(256, 3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(128, 5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.4)(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='Deep-LSTM')

def create_deep_inception_tdnn_model(input_shape, n_classes):
    inputs = Input(shape=input_shape)
    t1 = Conv1D(160, 3, padding='same', activation='relu')(inputs)
    t2 = Conv1D(160, 5, padding='same', activation='relu')(inputs)
    t3 = Conv1D(160, 7, padding='same', activation='relu')(inputs)
    x = concatenate([t1, t2, t3], axis=-1)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Conv1D(256, 1, padding='same', activation='relu', name='last_conv_layer')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='Deep-TDNN')

def gated_conv_block(input_tensor, filters, kernel_size):
    gate = Conv1D(filters, kernel_size, padding='same', activation='sigmoid')(input_tensor)
    feature = Conv1D(filters, kernel_size, padding='same', activation='relu')(input_tensor)
    return multiply([gate, feature])

def create_deep_gated_cnn_model(input_shape, n_classes):
    inputs = Input(shape=input_shape)
    x = gated_conv_block(inputs, 128, 3)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = gated_conv_block(x, 256, 7)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='Deep-Gated-CNN')


# --- 5. PART 1: CROSS-VALIDATION & PERFORMANCE EVALUATION ---
print(f"--- Step 2: Running {N_SPLITS}-Fold Cross-Validation ---")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof_preds = {}
fold_accuracies = []

base_models_fns = [
    create_deep_dnn_model,
    create_deep_conv_lstm_model,
    create_deep_inception_tdnn_model,
    create_deep_gated_cnn_model
]


# Initialize dictionaries for out-of-fold predictions
base_model_names = ['Deep-DNN', 'Deep-LSTM', 'Deep-TDNN', 'Deep-Gated-CNN']
for name in base_model_names:
    oof_preds[name] = np.zeros((len(X), N_CLASSES))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n===== FOLD {fold + 1}/{N_SPLITS} =====")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

    # Preprocessing
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
    X_val_scaled = scaler.transform(imputer.transform(X_val))

    # Reshape for sequence models if necessary
    X_train_seq = np.reshape(X_train_scaled, (X_train_scaled.shape[0], TIMESTEPS, FEATURES_PER_TIMESTEP))
    X_val_seq = np.reshape(X_val_scaled, (X_val_scaled.shape[0], TIMESTEPS, FEATURES_PER_TIMESTEP))

    y_train_cat = to_categorical(y_train_fold, num_classes=N_CLASSES)
    y_val_cat = to_categorical(y_val_fold, num_classes=N_CLASSES)

    for model_fn in base_models_fns:
        # --- This is the corrected line ---
        is_dnn = model_fn.__name__ == 'create_deep_dnn_model'

        input_shape_dnn = (X_train_scaled.shape[1],)
        input_shape_seq = (TIMESTEPS, FEATURES_PER_TIMESTEP)

        model = model_fn(input_shape_dnn if is_dnn else input_shape_seq, N_CLASSES)
        train_data = X_train_scaled if is_dnn else X_train_seq
        val_data = X_val_scaled if is_dnn else X_val_seq
        
        print(f"--- Training {model.name} ---")
        early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0)
        
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_data, y_train_cat, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_data=(val_data, y_val_cat), callbacks=[early_stopping, reduce_lr], verbose=0)

        val_preds = model.predict(val_data, batch_size=BATCH_SIZE * 2, verbose=0)
        oof_preds[model.name][val_idx] = val_preds
        acc = accuracy_score(np.argmax(y_val_cat, axis=1), np.argmax(val_preds, axis=1))
        print(f"{model.name} Fold {fold+1} Val Accuracy: {acc:.2%}")
        fold_accuracies.append({'Fold': fold+1, 'Model': model.name, 'Accuracy': acc})
        
        tf.keras.backend.clear_session()

# --- 6. META-MODEL TRAINING & FINAL ACCURACY ---
print("\n--- Step 3: Training Stacking Meta-Model and Evaluating Performance ---")
meta_X = np.concatenate([oof_preds[name] for name in sorted(oof_preds.keys())], axis=1)

meta_model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                                loss_function='MultiClass', random_seed=RANDOM_STATE, verbose=False)
meta_model.fit(meta_X, y)
predicted_labels = np.argmax(meta_model.predict_proba(meta_X), axis=1)
stacking_accuracy = accuracy_score(y, predicted_labels)

# Display final results
results_df = pd.DataFrame(fold_accuracies).groupby('Model')['Accuracy'].mean().sort_values(ascending=False)
print("\n--- Final Performance Results ---")
print("Average Accuracy of Base Models Across Folds:")
print(results_df.map('{:.2%}'.format).to_string())
print(f"\nStacking Ensemble Final OOF Accuracy (CatBoost): {stacking_accuracy:.2%}")

# --- 7. PART 2: FINAL MODEL TRAINING ON FULL DATASET ---
print("\n--- Step 4: Training Final Models on Full Dataset for Explainability ---")

# Preprocess the entire dataset
final_imputer = SimpleImputer(strategy='median')
final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(final_imputer.fit_transform(X))
X_seq = np.reshape(X_scaled, (X_scaled.shape[0], TIMESTEPS, FEATURES_PER_TIMESTEP))
y_cat = to_categorical(y, num_classes=N_CLASSES)

final_models = {}
for model_fn in base_models_fns:
    # --- This is the corrected line ---
    is_dnn = model_fn.__name__ == 'create_deep_dnn_model'
    input_shape_dnn = (X_scaled.shape[1],)
    input_shape_seq = (TIMESTEPS, FEATURES_PER_TIMESTEP)
    
    model = model_fn(input_shape_dnn if is_dnn else input_shape_seq, N_CLASSES)
    train_data = X_scaled if is_dnn else X_seq
    
    print(f"Training final {model.name} model...")
    # Using fewer epochs for final training as an example, adjust as needed
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, y_cat, epochs=50, batch_size=BATCH_SIZE, verbose=0, callbacks=[ReduceLROnPlateau(patience=10)])
    final_models[model.name] = model

# --- 8. PART 3: EXPLAINABILITY ANALYSIS (XAI) ---
print("\n--- Step 5: Generating Explanations for Final Models ---")

# Select a single instance to explain (e.g., the 10th sample in the dataset)
instance_idx = 10
instance_raw = X[instance_idx:instance_idx+1]
instance_scaled = final_scaler.transform(final_imputer.transform(instance_raw))
instance_dnn = instance_scaled
instance_seq = np.reshape(instance_scaled, (1, TIMESTEPS, FEATURES_PER_TIMESTEP))
true_label = CLASS_NAMES[y[instance_idx]]
print(f"Generating explanations for instance #{instance_idx}, True Label: '{true_label}'\n")

# A. Explain the Meta-Model (CatBoost) with SHAP
print("... Explaining CatBoost Meta-Model with SHAP")
meta_explainer = shap.TreeExplainer(meta_model)
shap_values_meta = meta_explainer.shap_values(meta_X)

base_model_names = sorted(oof_preds.keys())
meta_feature_names = [f'{name}_prob_class_{i}' for name in base_model_names for i in range(N_CLASSES)]

plt.figure()
shap.summary_plot(shap_values_meta, features=meta_X, feature_names=meta_feature_names,
                  class_names=CLASS_NAMES, plot_type="bar", max_display=15, show=False)
plt.title('SHAP Feature Importance for CatBoost Meta-Model', fontsize=14)
plt.tight_layout()
plt.savefig('xai_meta_model_shap_summary.png', dpi=300)
plt.show()

# B. Explain the DNN Model with LIME
print("... Explaining DNN Model with LIME")
dnn_predict_fn = lambda x: final_models['Deep-DNN'].predict(x)
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_scaled,
    feature_names=df.drop(columns=[TARGET_COLUMN]).columns,
    class_names=CLASS_NAMES,
    mode='classification'
)
lime_explanation = lime_explainer.explain_instance(
    instance_dnn[0], dnn_predict_fn, num_features=10, top_labels=1
)
# LIME plots are best viewed in a notebook or saved to HTML
lime_explanation.save_to_file('xai_dnn_lime_explanation.html')
print("LIME explanation for DNN saved to 'xai_dnn_lime_explanation.html'")


# C. Explain the LSTM Model with SHAP GradientExplainer
print("... Explaining LSTM Model with SHAP")
background_data = X_seq[np.random.choice(X_seq.shape[0], 100, replace=False)]
lstm_explainer = shap.GradientExplainer(final_models['Deep-LSTM'], background_data)
shap_values_lstm = lstm_explainer.shap_values(instance_seq)

# Sum SHAP values across timesteps for overall feature importance
# This is the corrected version
# For binary classification, SHAP returns one set of values. Access it at index 0.
# --- This is the corrected calculation ---

# 1. Sum over the timesteps (axis=0), resulting in shape (18, 2)
multi_class_shap_sum = np.sum(np.abs(shap_values_lstm[0]).squeeze(), axis=0)

# 2. Sum over the classes (axis=1) to get a single importance value per feature
shap_summed = multi_class_shap_sum.sum(axis=1)
feature_names_ts = [f'F{i+1}' for i in range(FEATURES_PER_TIMESTEP)]
top_n = 10
indices = np.argsort(shap_summed)[-top_n:]

plt.figure(figsize=(10, 6))
# --- This is the corrected plotting block ---

# Get the labels and values for the top features
y_labels = np.array(feature_names_ts)[indices]
x_values = shap_summed[indices]

# Convert the numpy array of labels to a standard Python list
plt.barh(y_labels.tolist(), x_values, color='skyblue')

# --- End of corrected block ---
plt.title(f'SHAP Feature Importance for Deep-LSTM (Instance #{instance_idx})', fontsize=14)
plt.xlabel("Sum of Absolute SHAP Values Across Timesteps")
plt.tight_layout()
plt.savefig('xai_lstm_shap_importance.png', dpi=300)
plt.show()

# D. Explain the TDNN (CNN) Model with Grad-CAM
print("... Explaining TDNN Model with Grad-CAM")
tdnn_model = final_models['Deep-TDNN']
pred_class_idx = np.argmax(tdnn_model.predict(instance_seq, verbose=0))
score = CategoricalScore(pred_class_idx)

# Find the last convolutional layer automatically
try:
    last_conv_layer_name = [layer.name for layer in tdnn_model.layers if 'conv1d' in layer.name.lower()][-1]
    print(f"Found last Conv1D layer for Grad-CAM: '{last_conv_layer_name}'")
    
    gradcam = Gradcam(tdnn_model, model_modifier=None, clone=True)
    cam_map = gradcam(score, instance_seq, penultimate_layer=last_conv_layer_name)[0]

    fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True, gridspec_kw={'height_ratios': [1, 3]})
    fig.suptitle(f'Grad-CAM for Deep-TDNN (Predicted: {CLASS_NAMES[pred_class_idx]}, True: {true_label})', fontsize=16)
    
    # Plot heatmap
    axes[0].imshow(np.expand_dims(cam_map, axis=0), cmap='jet', aspect='auto')
    axes[0].set_yticks([])
    axes[0].set_title('Grad-CAM Heatmap (Timestep Importance)')
    
    # Plot original data
    axes[1].imshow(instance_seq.squeeze().T, cmap='viridis', aspect='auto')
    axes[1].set_title('Original Input Sequence Data')
    axes[1].set_ylabel('Features per Timestep')
    axes[1].set_xlabel('Timesteps')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('xai_tdnn_gradcam.png', dpi=300)
    plt.show()

except IndexError:
    print("Could not automatically find a Conv1D layer for Grad-CAM.")
except Exception as e:
    print(f"An error occurred during Grad-CAM generation: {e}")

print("\n✅ Process complete. Explainability plots have been saved to disk.")