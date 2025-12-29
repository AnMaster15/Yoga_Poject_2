import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report

# --------------------
# 1. Advanced Feature Engineering Function
# --------------------
def create_advanced_features(data):
    """
    Creates advanced features, including log transforms for skewed data.
    """
    print("Creating advanced features...")
    df_feat = data.copy()
    FEATURES = ['X1', 'X2', 'X3', 'X4', 'X5']
    
    # --- Log transform the highly skewed features FIRST ---
    df_feat['X3_log'] = np.log1p(df_feat['X3'])
    df_feat['X4_log'] = np.log1p(df_feat['X4'])
    
    # --- Base Features (Lags, Rolling, Diffs) ---
    features_to_engineer = FEATURES + ['X3_log', 'X4_log']
    
    for col in features_to_engineer:
        df_feat[f'{col}_lag_1'] = df_feat[col].shift(1)
        df_feat[f'{col}_roll_mean_10'] = df_feat[col].rolling(window=10).mean()
        df_feat[f'{col}_roll_std_10'] = df_feat[col].rolling(window=10).std()
        df_feat[f'{col}_diff_1'] = df_feat[col].diff(periods=1)

    # --- Deviation from Rolling Mean ---
    for col in FEATURES:
        df_feat[f'{col}_deviation_from_mean_30'] = df_feat[col] / (df_feat[col].rolling(window=30).mean() + 1e-6)
        
    # --- Interaction Feature ---
    df_feat['X4_log_x_X5'] = df_feat['X4_log'] * df_feat['X5']
    
    return df_feat


# --------------------
# 2. Load and Prepare Data
# --------------------
print("Loading data...")
# Make sure to update the path to your data files
df = pd.read_parquet("train.parquet")
test_df = pd.read_parquet("test.parquet")

# Ensure data is sorted chronologically
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
test_df['Date'] = pd.to_datetime(test_df['Date'])
test_df = test_df.sort_values('Date').reset_index(drop=True)

# Apply the new feature engineering function
df_featured = create_advanced_features(df)

TARGET = 'target'
DROP_COLS_FINAL = [TARGET, 'Date', 'ID']
X = df_featured.drop(columns=[TARGET, 'Date'])
y = df_featured[TARGET].astype(int)

# --------------------
# 3. Train Base Models and Generate Predictions
# --------------------
N_SPLITS = 5
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# --- Define Base Models ---
scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]
lgbm = lgb.LGBMClassifier(class_weight='balanced', random_state=42, n_estimators=750, learning_rate=0.02)
xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, n_estimators=750, learning_rate=0.02, eval_metric='logloss')

base_models = {
    'lgbm': lgbm,
    'xgb': xgb_model
}

# Dictionaries to store predictions from each model
oof_predictions = {}
test_predictions = {}

print(f"\nStarting training for base models...")
for name, model in base_models.items():
    print(f"--- Training {name} ---")
    
    oof_preds = np.zeros(len(df))
    test_preds = np.zeros(len(test_df))
    
    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        print(f"  Fold {fold+1}/{N_SPLITS}")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        X_train = X_train.bfill().ffill()
        X_val = X_val.bfill().ffill()
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # ############# CORRECTED SECTION #############
        # The 'verbose' argument is removed to prevent the error
        model.fit(X_train_scaled, y_train)
        # ############# END OF CORRECTION #############
        
        oof_preds[val_index] = model.predict_proba(X_val_scaled)[:, 1]
        
        history_df = df.iloc[train_index].tail(100)
        combined_df = pd.concat([history_df, test_df])
        combined_featured = create_advanced_features(combined_df)
        
        X_sub = combined_featured.iloc[-len(test_df):].drop(columns=DROP_COLS_FINAL, errors='ignore')
        X_sub = X_sub.bfill().ffill()
        X_sub = X_sub[X_train.columns]
        
        X_sub_scaled = scaler.transform(X_sub)
        test_preds += model.predict_proba(X_sub_scaled)[:, 1] / N_SPLITS
        
    oof_predictions[name] = oof_preds
    test_predictions[name] = test_preds

print("\n--- Base model training complete ---")

# --------------------
# 4. Find Best Blend and Threshold
# --------------------
print("Finding best blend weight and threshold...")
# We only use the validation fold predictions for finding the best blend
valid_indices = oof_predictions['lgbm'] != 0
oof_lgbm = oof_predictions['lgbm'][valid_indices]
oof_xgb = oof_predictions['xgb'][valid_indices]
y_true = y[valid_indices]

weights = np.linspace(0, 1, 101)
best_f1 = 0
best_weight = 0
best_thresh = 0

for w in weights:
    blended_oof_preds = (w * oof_lgbm) + ((1 - w) * oof_xgb)
    
    thresholds = np.linspace(0.1, 0.9, 100)
    f1_scores = [f1_score(y_true, (blended_oof_preds > t).astype(int)) for t in thresholds]
    
    current_best_f1 = max(f1_scores)
    
    if current_best_f1 > best_f1:
        best_f1 = current_best_f1
        best_weight = w
        best_thresh = thresholds[np.argmax(f1_scores)]

print(f"\nBest Blended F1 Score on OOF: {best_f1:.4f}")
print(f"  - Best Weight for LGBM: {best_weight:.2f}")
print(f"  - Best Weight for XGB: {1-best_weight:.2f}")
print(f"  - Best Threshold for Blend: {best_thresh:.4f}")

# --------------------
# 5. Generate Submission File
# --------------------
print("\nGenerating final submission file...")
blended_test_preds = (best_weight * test_predictions['lgbm']) + ((1 - best_weight) * test_predictions['xgb'])
final_predictions = (blended_test_preds > best_thresh).astype(int)

submission = pd.DataFrame({'ID': test_df['ID'], 'target': final_predictions})
submission.to_csv("submission_blended.csv", index=False)

print("\nSubmission file 'submission_blended.csv' saved.")
print("Final prediction breakdown:")
print(submission['target'].value_counts())