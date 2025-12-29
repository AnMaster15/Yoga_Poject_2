import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

def load_data(file_path, target_col='stress_score_weighted'):
    """Load and prepare data from a CSV file with improved error handling"""
    # Load data
    try:
        # Try to read CSV with pandas, which will handle the header row correctly
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data with shape: {df.shape}")
        
        # Print first few rows to help diagnose issues
        print("\nFirst 5 rows of data:")
        print(df.head())
        
        # Print column names to help identify issues
        print("\nColumn names:")
        print(df.columns.tolist())
        
        # Remove unnecessary columns
        columns_to_remove = ['Segment', 'Condition', 'Condition_Name', 'Start_Sample', 'End_Sample']
        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(columns=[col])
                print(f"Removed column: {col}")
        
        # Check for the target column
        if target_col not in df.columns:
            # If target column not found, check if there's a similar column
            possible_targets = [col for col in df.columns if 'stress' in col.lower()]
            if possible_targets:
                print(f"\nTarget column '{target_col}' not found, but found these possible stress-related columns:")
                print(possible_targets)
                target_col = possible_targets[0]  # Use the first match
                print(f"Using '{target_col}' as the target column")
            else:
                raise ValueError(f"Target column '{target_col}' not found in dataset and no stress-related columns found")
        
        # Check for non-numeric columns
        non_numeric_cols = []
        for col in df.columns:
            try:
                pd.to_numeric(df[col])
            except:
                non_numeric_cols.append(col)
        
        if non_numeric_cols:
            print(f"\nWarning: Found non-numeric columns: {non_numeric_cols}")
            print("These columns will be dropped for modeling except Subject_ID")
            
            # Drop non-numeric columns except the target and Subject_ID
            for col in non_numeric_cols:
                if col != target_col and col != 'Subject_ID':
                    df = df.drop(columns=[col])
        
        # Handle the target column
        try:
            # Try to convert target to numeric
            df[target_col] = pd.to_numeric(df[target_col])
        except:
            print(f"\nError: Target column '{target_col}' contains non-numeric values")
            print("Sample values from target column:")
            print(df[target_col].head(10).tolist())
            raise ValueError(f"Target column '{target_col}' must be numeric")
        
        # Keep Subject_ID for later analysis but don't include in modeling features
        subject_ids = None
        if 'Subject_ID' in df.columns:
            subject_ids = df['Subject_ID']
            X = df.drop(columns=[target_col, 'Subject_ID'])
        else:
            X = df.drop(columns=[target_col])
        
        y = df[target_col]
        
        # Convert all remaining columns to numeric (just to be safe)
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Handle any NaN values created by coercion
        X = X.fillna(X.mean())
        
        print(f"\nFeatures shape after preprocessing: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        return X, y, subject_ids
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def preprocess_data(X, y, subject_ids=None, test_size=0.2, scaler_type='standard', select_k_features=None):
    """Preprocess the data: split, scale, and optionally select features"""
    # Split the data
    if subject_ids is not None:
        X_train, X_test, y_train, y_test, subject_train, subject_test = train_test_split(
            X, y, subject_ids, test_size=test_size, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)
        subject_train, subject_test = None, None
    
    # Scale features
    if scaler_type == 'robust':
        scaler = RobustScaler()  # Less influenced by outliers
    else:
        scaler = StandardScaler()  # Standard z-score normalization
        
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Optional feature selection
    feature_selector = None
    if select_k_features is not None and select_k_features < X.shape[1]:
        feature_selector = SelectKBest(f_regression, k=select_k_features)
        X_train_scaled = feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_scaled = feature_selector.transform(X_test_scaled)
        
        # Get selected feature names
        selected_indices = feature_selector.get_support(indices=True)
        selected_features = X.columns[selected_indices]
        print(f"Selected top {select_k_features} features: {selected_features.tolist()}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_selector, subject_train, subject_test

def build_model(input_dim, architecture=[128, 64, 32], dropout_rate=0.3,
               learning_rate=0.001, activation='leaky_relu'):
    """Build a neural network model for stress prediction"""
    model = Sequential()
    
    # Input layer
    if activation == 'leaky_relu':
        model.add(Dense(architecture[0], input_dim=input_dim))
        model.add(LeakyReLU(alpha=0.1))
    else:
        model.add(Dense(architecture[0], input_dim=input_dim, activation=activation))
        
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Hidden layers
    for units in architecture[1:]:
        if activation == 'leaky_relu':
            model.add(Dense(units))
            model.add(LeakyReLU(alpha=0.1))
        else:
            model.add(Dense(units, activation=activation))
            
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    
    # Output layer for regression
    model.add(Dense(1, activation='linear'))
    
    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Display model summary
    model.summary()
    
    return model

def train_model(model, X_train, y_train, epochs=200, batch_size=32, validation_split=0.2,
               patience=25, verbose=1, model_path=None):
    """Train the model with early stopping and learning rate reduction"""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
    ]
    
    # Add model checkpoint if path provided
    if model_path:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        callbacks.append(ModelCheckpoint(
            f'{model_path}/best_model.h5', save_best_only=True, monitor='val_loss'
        ))
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=verbose
    )
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

def evaluate_model(model, X_test, y_test, subject_ids=None):
    """Evaluate the model on test data, with subject-wise analysis if available"""
    # Make predictions
    y_pred = model.predict(X_test).flatten()
    
    # Calculate overall metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }
    
    # Print overall metrics
    print("\nOverall Model Evaluation:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Subject-wise analysis if subject IDs are available
    if subject_ids is not None:
        print("\nSubject-wise Regression Analysis:")
        
        # Create a DataFrame with predictions, true values, and subject IDs
        results_df = pd.DataFrame({
            'Subject_ID': subject_ids,
            'Actual': y_test,
            'Predicted': y_pred
        })
        
        # Group by subject and calculate metrics for each
        subject_metrics = {}
        unique_subjects = results_df['Subject_ID'].unique()
        
        for subject in unique_subjects:
            subject_data = results_df[results_df['Subject_ID'] == subject]
            subject_actual = subject_data['Actual']
            subject_pred = subject_data['Predicted']
            
            if len(subject_actual) > 1:  # Need at least 2 points for most metrics
                subject_mse = mean_squared_error(subject_actual, subject_pred)
                subject_rmse = np.sqrt(subject_mse)
                subject_mae = mean_absolute_error(subject_actual, subject_pred)
                
                # R² can be negative if the model is worse than the mean baseline
                try:
                    subject_r2 = r2_score(subject_actual, subject_pred)
                except:
                    subject_r2 = float('nan')
                
                subject_metrics[subject] = {
                    'MSE': subject_mse,
                    'RMSE': subject_rmse,
                    'MAE': subject_mae,
                    'R²': subject_r2,
                    'Count': len(subject_actual)
                }
        
        # Convert to DataFrame for easier display
        subject_metrics_df = pd.DataFrame(subject_metrics).T
        print(subject_metrics_df)
        
        # Plot subject-wise R² 
        plt.figure(figsize=(12, 6))
        plt.bar(subject_metrics_df.index.astype(str), subject_metrics_df['R²'])
        plt.title('R² Score by Subject')
        plt.xlabel('Subject ID')
        plt.ylabel('R² Score')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        
    # Plot predictions
    plot_predictions(y_test, y_pred)
        
    return metrics, y_pred

def feature_importance(X, y):
    """Calculate feature importance using correlation with target"""
    # Create a dataframe with features and target
    df = pd.concat([X, pd.Series(y, name='target')], axis=1)
    
    # Calculate correlation with target
    correlations = df.corr()['target'].drop('target').abs().sort_values(ascending=False)
    
    # Display top correlations
    print("\nTop feature correlations with stress score:")
    print(correlations.head(10))
    
    # Plot top 15 correlations
    plt.figure(figsize=(12, 8))
    top_15 = correlations.head(15)
    sns.barplot(x=top_15.values, y=top_15.index)
    plt.title('Top 15 Features by Correlation with Stress Score')
    plt.xlabel('Absolute Correlation')
    plt.tight_layout()
    plt.show()
    
    return correlations

def cross_validate(X, y, n_splits=5, epochs=100, batch_size=32):
    """Perform k-fold cross-validation"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    
    print(f"\nPerforming {n_splits}-fold cross-validation:")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold+1}/{n_splits}")
        
        # Split data
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Preprocess
        scaler = RobustScaler()  # Use RobustScaler for cross-validation
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        
        # Build model for this fold
        input_dim = X_train_scaled.shape[1]
        model = build_model(input_dim)
        
        # Training with early stopping
        callbacks = [EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]
        model.fit(
            X_train_scaled, y_train_fold,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_scaled, y_val_fold),
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        y_pred = model.predict(X_val_scaled).flatten()
        metrics = {
            'MSE': mean_squared_error(y_val_fold, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_val_fold, y_pred)),
            'MAE': mean_absolute_error(y_val_fold, y_pred),
            'R²': r2_score(y_val_fold, y_pred)
        }
        fold_metrics.append(metrics)
        print(f"Fold {fold+1} metrics: MSE={metrics['MSE']:.4f}, R²={metrics['R²']:.4f}")
    
    # Calculate average metrics
    avg_metrics = {
        metric: np.mean([fold[metric] for fold in fold_metrics])
        for metric in fold_metrics[0].keys()
    }
    
    std_metrics = {
        metric: np.std([fold[metric] for fold in fold_metrics])
        for metric in fold_metrics[0].keys()
    }
    
    print("\nCross-validation results:")
    for metric in avg_metrics.keys():
        print(f"{metric}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")
        
    # Plot cross-validation results
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['MSE', 'MAE', 'R²']
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(1, 3, i+1)
        values = [fold[metric] for fold in fold_metrics]
        plt.bar(range(1, n_splits+1), values)
        plt.title(f'{metric} by Fold')
        plt.xlabel('Fold')
        plt.ylabel(metric)
    plt.tight_layout()
    plt.show()
        
    return {
        'fold_metrics': fold_metrics,
        'avg_metrics': avg_metrics,
        'std_metrics': std_metrics
    }

def save_model(model, scaler, feature_selector, feature_names, model_path='stress_model'):
    """Save the model, scaler, and feature selector to disk"""
    # Create directory if it doesn't exist
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    # Save model
    model.save(f'{model_path}/model.h5')
    
    # Save scaler
    if scaler is not None:
        joblib.dump(scaler, f'{model_path}/scaler.pkl')
        
    # Save feature selector
    if feature_selector is not None:
        joblib.dump(feature_selector, f'{model_path}/feature_selector.pkl')
        
    # Save feature names
    if feature_names is not None:
        pd.Series(feature_names).to_csv(f'{model_path}/feature_names.csv', index=False)
        
    print(f"Model and preprocessors saved to {model_path}/")
    
    return f"Model saved to {model_path}/"

def load_saved_model(model_path='stress_model'):
    """Load the model, scaler, and feature selector from disk"""
    try:
        # Load model
        model = load_model(f'{model_path}/model.h5')
        print("Model loaded successfully")
        
        # Load scaler
        scaler = None
        scaler_path = f'{model_path}/scaler.pkl'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("Scaler loaded successfully")
            
        # Load feature selector
        feature_selector = None
        selector_path = f'{model_path}/feature_selector.pkl'
        if os.path.exists(selector_path):
            feature_selector = joblib.load(selector_path)
            print("Feature selector loaded successfully")
            
        # Load feature names
        feature_names = None
        feature_names_path = f'{model_path}/feature_names.csv'
        if os.path.exists(feature_names_path):
            feature_names = pd.read_csv(feature_names_path).iloc[:, 0].values
            print(f"Loaded {len(feature_names)} feature names")
            
        return model, scaler, feature_selector, feature_names
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None

def predict_stress(model, new_data, scaler=None, feature_selector=None, feature_names=None):
    """Make stress predictions on new data"""
    # Convert to DataFrame if array
    if not isinstance(new_data, pd.DataFrame) and feature_names is not None:
        new_data = pd.DataFrame(new_data, columns=feature_names)
        
    # Ensure we have the right features
    if isinstance(new_data, pd.DataFrame) and feature_names is not None:
        missing_features = set(feature_names) - set(new_data.columns)
        if missing_features:
            print(f"Warning: Missing features in input data: {missing_features}")
            return None
            
        # Reorder columns to match training data
        new_data = new_data[feature_names]
        
    # Preprocess data
    if scaler is not None:
        new_data = scaler.transform(new_data)
        
    if feature_selector is not None:
        new_data = feature_selector.transform(new_data)
        
    # Make predictions
    predictions = model.predict(new_data).flatten()
    
    # Display summary statistics of predictions
    print("\nPrediction Results:")
    print(f"Mean predicted stress: {predictions.mean():.2f}")
    print(f"Min predicted stress: {predictions.min():.2f}")
    print(f"Max predicted stress: {predictions.max():.2f}")
    
    # Plot histogram of predictions
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, bins=20, alpha=0.7, color='blue')
    plt.title('Distribution of Predicted Stress Scores')
    plt.xlabel('Predicted Stress Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return predictions

def plot_training_history(history):
    """Plot the training history"""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred):
    """Plot the predicted vs actual values"""
    plt.figure(figsize=(10, 6))
    
    # Plot actual vs predicted
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor='k', s=50)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    # Calculate metrics for display
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Add labels and metrics
    plt.title(f'Actual vs Predicted Stress Scores\nMSE: {mse:.4f}, R²: {r2:.4f}')
    plt.xlabel('Actual Stress Score')
    plt.ylabel('Predicted Stress Score')
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    plt.annotate(f'MSE: {mse:.4f}', xy=(0.05, 0.95), xycoords='axes fraction')
    plt.annotate(f'R²: {r2:.4f}', xy=(0.05, 0.90), xycoords='axes fraction')
    
    plt.tight_layout()
    plt.show()

def tune_hyperparameters(X, y, param_grid):
    """Simple hyperparameter tuning for the neural network model"""
    # Split data once for consistent evaluation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale data
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Initialize results
    results = []
    best_val_loss = float('inf')
    best_params = None
    best_model = None
    
    # Create parameter combinations
    def create_param_combinations(param_grid):
        keys = param_grid.keys()
        combinations = []
        
        # Recursive function to generate combinations
        def recurse(combination, keys_list):
            if not keys_list:
                combinations.append(combination.copy())
                return
            
            current_key = keys_list[0]
            for value in param_grid[current_key]:
                combination[current_key] = value
                recurse(combination, keys_list[1:])
        
        recurse({}, list(keys))
        return combinations
    
    # Get all parameter combinations
    param_combinations = create_param_combinations(param_grid)
    print(f"Testing {len(param_combinations)} hyperparameter combinations...")
    
    for i, params in enumerate(param_combinations):
        print(f"\nTesting combination {i+1}/{len(param_combinations)}: {params}")
        
        # Build model with current parameters
        input_dim = X_train_scaled.shape[1]
        model = Sequential()
        
        # Input layer
        if params.get('activation', 'relu') == 'leaky_relu':
            model.add(Dense(params['first_layer'], input_dim=input_dim))
            model.add(LeakyReLU(alpha=0.1))
        else:
            model.add(Dense(params['first_layer'], input_dim=input_dim, 
                           activation=params.get('activation', 'relu')))
            
        model.add(BatchNormalization())
        model.add(Dropout(params.get('dropout_rate', 0.3)))
        
        # Hidden layers (if specified)
        if 'second_layer' in params and params['second_layer'] > 0:
            if params.get('activation', 'relu') == 'leaky_relu':
                model.add(Dense(params['second_layer']))
                model.add(LeakyReLU(alpha=0.1))
            else:
                model.add(Dense(params['second_layer'], 
                               activation=params.get('activation', 'relu')))
                
            model.add(BatchNormalization())
            model.add(Dropout(params.get('dropout_rate', 0.3)))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        optimizer = Adam(learning_rate=params.get('learning_rate', 0.001))
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        # Train with early stopping
        callbacks = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]
        history = model.fit(
            X_train_scaled, y_train,
            epochs=100,  # Max epochs
            batch_size=params.get('batch_size', 32),
            validation_data=(X_val_scaled, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate on validation set
        val_loss = model.evaluate(X_val_scaled, y_val, verbose=0)[0]
        y_pred = model.predict(X_val_scaled).flatten()
        val_r2 = r2_score(y_val, y_pred)
        
        print(f"Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}")
        
        # Save results
        results.append({
            'params': params,
            'val_loss': val_loss,
            'val_r2': val_r2,
            'epochs_trained': len(history.history['loss'])
        })
        
        # Check if this is best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            best_model = model
            print(f"New best model found!")
    
    # Sort results by validation loss
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('val_loss')
    
    print("\nTop 5 hyperparameter combinations:")
    print(results_df.head(5)[['params', 'val_loss', 'val_r2']])
    
    print(f"\nBest hyperparameters: {best_params}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Plot top combinations
    plt.figure(figsize=(12, 6))
    top_5 = results_df.head(5)
    plt.subplot(1, 2, 1)
    plt.bar(range(5), top_5['val_loss'])
    plt.title('Validation Loss - Top 5')
    plt.xlabel('Combination Rank')
    plt.ylabel('MSE')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(5), top_5['val_r2'])
    plt.title('Validation R² - Top 5')
    plt.xlabel('Combination Rank')
    plt.ylabel('R²')
    
    plt.tight_layout()
    plt.show()
    
    return best_model, best_params, results_df

# Example usage
def example_workflow(file_path, target_col='stress_score_weighted'):
    """Example of how to use the functions in sequence"""
    # 1. Load data
    X, y, subject_ids = load_data(file_path, target_col=target_col)
    
    # 2. Preprocess data
    X_train, X_test, y_train, y_test, scaler, feature_selector, subject_train, subject_test = preprocess_data(
        X, y, subject_ids, scaler_type='robust'
    )
    
    # 3. Build model
    model = build_model(input_dim=X_train.shape[1])
    
    # 4. Train model
    trained_model, history = train_model(model, X_train, y_train)
    
    # 5. Evaluate model
    metrics, predictions = evaluate_model(trained_model, X_test, y_test, subject_test)
    
    # 6. Save model if needed
    save_model(trained_model, scaler, feature_selector, X.columns.tolist())
    
    return trained_model, metrics, predictions

if __name__ == "__main__":
    # Specify the path to your dataset file
    data_file = "Final_merged_dataset_with_weighted_stress.csv"
    
    # If you have a specific column for stress scores, specify it here
    target_column = "stress_score_weighted"
    
    try:
        # Run the full workflow
        print("Starting stress prediction analysis...")
        model, metrics, predictions = example_workflow(data_file, target_column)
        print("\nAnalysis completed successfully!")
        
        # Print final performance metrics
        print("\nFinal model performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
    except FileNotFoundError:
        print("\nERROR: Data file not found. Please update the 'data_file' variable with the correct path.")
        print("You can also try creating a sample dataset for testing:")
        
        # Create sample data for demonstration
        print("\nCreating sample data for demonstration...")
        np.random.seed(42)
        sample_size = 500
        n_features = 10
        
        # Generate random features
        sample_X = pd.DataFrame(
            np.random.randn(sample_size, n_features),
            columns=[f'feature_{i+1}' for i in range(n_features)]
        )
        
        # Add Subject_ID column with random subjects
        num_subjects = 10
        subject_ids = np.random.randint(1, num_subjects + 1, size=sample_size)
        sample_X['Subject_ID'] = subject_ids
        
        # Generate target values with some relationship to features and subjects
        # Different subjects will have slightly different baseline stress levels
        subject_effect = pd.Series(subject_ids).map({i: np.random.uniform(-1, 1) for i in range(1, num_subjects + 1)})
        
        sample_y = 2 * sample_X['feature_1'] - 1.5 * sample_X['feature_2'] + \
                   0.5 * sample_X['feature_5'] + subject_effect + np.random.randn(sample_size) * 0.5
        
        # Create sample dataframe
        sample_df = sample_X.copy()
        sample_df[target_column] = sample_y
        
        print(f"Sample data created with shape: {sample_df.shape}")
        
        # Save sample data
        sample_file = "sample_stress_data.csv"
        sample_df.to_csv(sample_file, index=False)
        print(f"Sample data saved to {sample_file}")
        
        # Run analysis on sample data
        print("\nRunning analysis on sample data...")
        model, metrics, predictions = example_workflow(sample_file, target_column)
        
        print("\nAnalysis on sample data completed successfully!")
        
    except Exception as e:
        print(f"\nAn error occurred during analysis: {e}")
        
        # Provide options for debugging
        print("\nDebugging suggestions:")
        print("1. Check if your data file exists and is accessible")
        print("2. Verify that your data contains the expected target column")
        print("3. Inspect the first few rows of your data to ensure it's formatted correctly")
        print("4. Check for missing values or invalid data types in your dataset")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import os
from scipy import stats

def compare_yoga_stress(before_file, after_file, target_col='stress_score_weighted'):
    """
    Analyze and compare stress levels before and after yoga using two CSV files.
    
    Parameters:
    -----------
    before_file : str
        Path to the CSV file containing data before yoga
    after_file : str
        Path to the CSV file containing data after yoga
    target_col : str
        Name of the column containing stress scores
    
    Returns:
    --------
    dict
        Dictionary containing results of the analysis
    """
    print("=" * 80)
    print(f"YOGA STRESS REDUCTION ANALYSIS")
    print("=" * 80)
    
    # 1. Load both datasets
    print("\n1. Loading datasets...")
    
    try:
        # Load before yoga data
        X_before, y_before, subject_before = load_data(before_file, target_col)
        print(f"\nLoaded pre-yoga data: {X_before.shape[0]} samples with {X_before.shape[1]} features")
        
        # Load after yoga data
        X_after, y_after, subject_after = load_data(after_file, target_col)
        print(f"Loaded post-yoga data: {X_after.shape[0]} samples with {X_after.shape[1]} features")
        
        # Check if we have the same subjects in both datasets
        if subject_before is not None and subject_after is not None:
            before_subjects = set(subject_before.unique())
            after_subjects = set(subject_after.unique())
            common_subjects = before_subjects.intersection(after_subjects)
            
            print(f"\nFound {len(before_subjects)} subjects in pre-yoga data")
            print(f"Found {len(after_subjects)} subjects in post-yoga data")
            print(f"Common subjects in both datasets: {len(common_subjects)}")
        
        # 2. Basic statistics comparison
        print("\n2. Basic stress score statistics:")
        before_stats = {
            'mean': y_before.mean(),
            'median': y_before.median(),
            'std': y_before.std(),
            'min': y_before.min(),
            'max': y_before.max()
        }
        
        after_stats = {
            'mean': y_after.mean(),
            'median': y_after.median(),
            'std': y_after.std(),
            'min': y_after.min(),
            'max': y_after.max()
        }
        
        # Create a dataframe to display stats
        stats_df = pd.DataFrame({
            'Before Yoga': before_stats,
            'After Yoga': after_stats,
            'Difference': {k: before_stats[k] - after_stats[k] for k in before_stats},
            'Change %': {k: ((before_stats[k] - after_stats[k]) / before_stats[k] * 100) 
                         if before_stats[k] != 0 else 0 for k in before_stats}
        })
        
        print(stats_df)
        
        # 3. Statistical significance test
        print("\n3. Statistical significance test:")
        t_stat, p_value = stats.ttest_ind(y_before, y_after, equal_var=False)
        print(f"Independent t-test: t={t_stat:.4f}, p-value={p_value:.4f}")
        if p_value < 0.05:
            print("The difference in stress scores is statistically significant (p < 0.05)")
        else:
            print("The difference in stress scores is not statistically significant (p >= 0.05)")
        
        # 4. Visualize the difference
        visualize_stress_difference(y_before, y_after, subject_before, subject_after)
        
        # 5. Train models on both datasets
        print("\n5. Training stress prediction models for before and after yoga...")
        
        # Preprocess before yoga data
        X_train_before, X_test_before, y_train_before, y_test_before, scaler_before, fs_before, _, _ = preprocess_data(
            X_before, y_before, subject_before, test_size=0.2, scaler_type='robust'
        )
        
        # Preprocess after yoga data
        X_train_after, X_test_after, y_train_after, y_test_after, scaler_after, fs_after, _, _ = preprocess_data(
            X_after, y_after, subject_after, test_size=0.2, scaler_type='robust'
        )
        
        # Train model on before yoga data
        model_before = build_model(input_dim=X_train_before.shape[1], architecture=[64, 32, 16])
        model_before, _ = train_model(model_before, X_train_before, y_train_before, epochs=100, verbose=0)
        
        # Train model on after yoga data
        model_after = build_model(input_dim=X_train_after.shape[1], architecture=[64, 32, 16])
        model_after, _ = train_model(model_after, X_train_after, y_train_after, epochs=100, verbose=0)
        
        # Evaluate models
        print("\n6. Model evaluation:")
        
        # Evaluate before yoga model
        print("\nPre-yoga model performance:")
        metrics_before, _ = evaluate_model(model_before, X_test_before, y_test_before, None)
        
        # Evaluate after yoga model
        print("\nPost-yoga model performance:")
        metrics_after, _ = evaluate_model(model_after, X_test_after, y_test_after, None)
        
        # 7. Compare feature importance
        print("\n7. Comparing feature importance before and after yoga:")
        importance_before = feature_importance(X_before, y_before, top_n=10, plot=False)
        importance_after = feature_importance(X_after, y_after, top_n=10, plot=False)
        
        # Compare top features
        compare_feature_importance(importance_before, importance_after)
        
        # 8. Generate prediction on both datasets to compare results
        print("\n8. Comparing prediction distributions:")
        
        # Get predictions for test sets
        y_pred_before = model_before.predict(X_test_before).flatten()
        y_pred_after = model_after.predict(X_test_after).flatten()
        
        # Plot prediction distributions
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(y_pred_before, kde=True, color='red', alpha=0.6, label='Before Yoga')
        sns.histplot(y_pred_after, kde=True, color='green', alpha=0.6, label='After Yoga')
        plt.title('Distribution of Predicted Stress Scores')
        plt.xlabel('Predicted Stress Score')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=[y_pred_before, y_pred_after], palette=['red', 'green'])
        plt.xticks([0, 1], ['Before Yoga', 'After Yoga'])
        plt.title('Boxplot of Predicted Stress Scores')
        plt.ylabel('Predicted Stress Score')
        plt.tight_layout()
        plt.show()
        
        # 9. Summary of findings
        print("\n" + "=" * 80)
        print("SUMMARY OF FINDINGS")
        print("=" * 80)
        
        # Calculate stress reduction percentage
        stress_reduction = (before_stats['mean'] - after_stats['mean']) / before_stats['mean'] * 100
        print(f"\n1. Average stress reduction after yoga: {stress_reduction:.2f}%")
        
        # Statistical significance
        print(f"2. Statistical significance: {'Significant' if p_value < 0.05 else 'Not significant'} (p={p_value:.4f})")
        
        # Model accuracy 
        print(f"3. Stress prediction model R² - Before: {metrics_before['R²']:.4f}, After: {metrics_after['R²']:.4f}")
        
        # Key biomarkers
        print("\n4. Key stress indicators that changed after yoga:")
        top_changed_features = get_top_changed_features(importance_before, importance_after)
        for i, (feature, change) in enumerate(top_changed_features.items(), 1):
            print(f"   {i}. {feature}: {change:.2f}% change in importance")
        
        return {
            'before_stats': before_stats,
            'after_stats': after_stats,
            'p_value': p_value,
            'stress_reduction': stress_reduction,
            'model_metrics_before': metrics_before,
            'model_metrics_after': metrics_after,
            'top_changed_features': top_changed_features
        }
    
    except Exception as e:
        print(f"\nError in analysis: {e}")
        return None

def visualize_stress_difference(y_before, y_after, subject_before=None, subject_after=None):
    """Visualize the difference in stress levels before and after yoga"""
    plt.figure(figsize=(15, 10))
    
    # 1. Overall distribution comparison
    plt.subplot(2, 2, 1)
    sns.histplot(y_before, color='red', alpha=0.6, label='Before Yoga', kde=True)
    sns.histplot(y_after, color='green', alpha=0.6, label='After Yoga', kde=True)
    plt.title('Distribution of Stress Scores Before and After Yoga')
    plt.xlabel('Stress Score')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 2. Boxplot comparison
    plt.subplot(2, 2, 2)
    data = [y_before, y_after]
    labels = ['Before Yoga', 'After Yoga']
    sns.boxplot(data=data, palette=['red', 'green'])
    plt.title('Stress Score Comparison')
    plt.ylabel('Stress Score')
    plt.xticks([0, 1], labels)
    
    # 3. Subject-wise comparison if subject IDs are available
    if subject_before is not None and subject_after is not None:
        plt.subplot(2, 2, 3)
        
        # Get common subjects
        common_subjects = set(subject_before).intersection(set(subject_after))
        
        # Take a sample if too many subjects
        if len(common_subjects) > 10:
            common_subjects = list(common_subjects)[:10]
        
        # Calculate mean stress per subject
        before_means = []
        after_means = []
        subjects = []
        
        for subject in common_subjects:
            before_mean = y_before[subject_before == subject].mean()
            after_mean = y_after[subject_after == subject].mean()
            
            before_means.append(before_mean)
            after_means.append(after_mean)
            subjects.append(subject)
        
        # Plot
        barWidth = 0.35
        r1 = np.arange(len(subjects))
        r2 = [x + barWidth for x in r1]
        
        plt.bar(r1, before_means, color='red', width=barWidth, label='Before Yoga')
        plt.bar(r2, after_means, color='green', width=barWidth, label='After Yoga')
        
        plt.xlabel('Subject ID')
        plt.ylabel('Average Stress Score')
        plt.title('Subject-wise Stress Score Comparison')
        plt.xticks([r + barWidth/2 for r in range(len(subjects))], subjects)
        plt.legend()
    
    # 4. Difference visualization
    plt.subplot(2, 2, 4)
    
    # Calculate means
    before_mean = y_before.mean()
    after_mean = y_after.mean()
    
    # Calculate percentage change
    percent_change = (before_mean - after_mean) / before_mean * 100
    
    # Plot
    plt.bar(['Before Yoga', 'After Yoga'], [before_mean, after_mean], color=['red', 'green'])
    plt.title(f'Average Stress Reduction: {percent_change:.2f}%')
    plt.ylabel('Average Stress Score')
    
    # Add text with values
    plt.text(0, before_mean/2, f'{before_mean:.2f}', ha='center', va='center', color='white', fontweight='bold')
    plt.text(1, after_mean/2, f'{after_mean:.2f}', ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def feature_importance(X, y, top_n=10, plot=True):
    """Calculate feature importance using correlation with target"""
    # Create a dataframe with features and target
    df = pd.concat([X, pd.Series(y, name='target')], axis=1)
    
    # Calculate correlation with target
    correlations = df.corr()['target'].drop('target').abs().sort_values(ascending=False)
    
    if plot:
        # Display top correlations
        print(f"\nTop {top_n} feature correlations with stress score:")
        print(correlations.head(top_n))
        
        # Plot top correlations
        plt.figure(figsize=(12, 8))
        top_features = correlations.head(top_n)
        sns.barplot(x=top_features.values, y=top_features.index)
        plt.title(f'Top {top_n} Features by Correlation with Stress Score')
        plt.xlabel('Absolute Correlation')
        plt.tight_layout()
        plt.show()
    
    return correlations

def compare_feature_importance(importance_before, importance_after, top_n=10):
    """Compare feature importance before and after yoga"""
    # Get top features from both
    top_before = importance_before.head(top_n*2)  # Get more to ensure overlap
    top_after = importance_after.head(top_n*2)
    
    # Get common features
    common_features = set(top_before.index).intersection(set(top_after.index))
    
    # Create dataframe for comparison
    comparison = pd.DataFrame({
        'Before Yoga': {feature: importance_before[feature] for feature in common_features},
        'After Yoga': {feature: importance_after[feature] for feature in common_features}
    })
    
    # Calculate absolute and relative difference
    comparison['Absolute Diff'] = comparison['Before Yoga'] - comparison['After Yoga']
    comparison['Relative Diff (%)'] = (comparison['Absolute Diff'] / comparison['Before Yoga'] * 100)
    
    # Sort by absolute difference
    comparison = comparison.sort_values('Absolute Diff', ascending=False)
    
    # Display top changed features
    print("\nTop features with changed importance after yoga:")
    print(comparison.head(top_n))
    
    # Plot comparison
    plt.figure(figsize=(14, 8))
    
    # Select top N features for clarity
    plot_features = comparison.head(min(top_n, len(comparison)))
    
    # Reshape for grouped barplot
    plot_data = pd.melt(
        plot_features.reset_index(), 
        id_vars='index', 
        value_vars=['Before Yoga', 'After Yoga'],
        var_name='Period', 
        value_name='Importance'
    )
    
    # Create grouped barplot
    sns.barplot(x='index', y='Importance', hue='Period', data=plot_data, palette=['red', 'green'])
    plt.title('Feature Importance Comparison Before and After Yoga')
    plt.xlabel('Feature')
    plt.ylabel('Importance (Correlation with Stress)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='')
    plt.tight_layout()
    plt.show()
    
    return comparison

def get_top_changed_features(importance_before, importance_after, top_n=5):
    """Get features with the biggest changes in importance"""
    # Get common features
    common_features = set(importance_before.index).intersection(set(importance_after.index))
    
    # Calculate relative change
    changes = {}
    for feature in common_features:
        before_val = importance_before[feature]
        after_val = importance_after[feature]
        
        if before_val > 0:  # Avoid division by zero
            rel_change = (before_val - after_val) / before_val * 100
            changes[feature] = rel_change
    
    # Sort by absolute change value
    sorted_changes = {k: v for k, v in sorted(changes.items(), key=lambda item: abs(item[1]), reverse=True)}
    
    # Return top N changes
    return dict(list(sorted_changes.items())[:top_n])

def run_yoga_stress_analysis(before_file, after_file):
    """Run the full yoga stress analysis workflow"""
    try:
        # Perform analysis
        results = compare_yoga_stress(before_file, after_file)
        
        if results:
            # Generate a report
            generate_report(before_file, after_file, results)
            
            print("\nAnalysis completed successfully!")
            return results
        else:
            print("\nAnalysis failed. Please check the error messages above.")
            return None
            
    except Exception as e:
        print(f"\nError running yoga stress analysis: {e}")
        return None

def generate_report(before_file, after_file, results):
    """Generate a simple report of the findings"""
    # Create report directory if it doesn't exist
    report_dir = "yoga_stress_analysis_report"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    # Create report file
    report_file = f"{report_dir}/yoga_stress_reduction_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("YOGA STRESS REDUCTION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Pre-yoga data: {before_file}\n")
        f.write(f"Post-yoga data: {after_file}\n\n")
        
        f.write("STRESS SCORE STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Before yoga mean: {results['before_stats']['mean']:.4f}\n")
        f.write(f"After yoga mean: {results['after_stats']['mean']:.4f}\n")
        f.write(f"Difference: {results['before_stats']['mean'] - results['after_stats']['mean']:.4f}\n")
        f.write(f"Percent reduction: {results['stress_reduction']:.2f}%\n\n")
        
        f.write("STATISTICAL SIGNIFICANCE\n")
        f.write("-" * 30 + "\n")
        f.write(f"p-value: {results['p_value']:.4f}\n")
        f.write(f"Significance: {'Significant (p<0.05)' if results['p_value'] < 0.05 else 'Not significant (p>=0.05)'}\n\n")
        
        f.write("KEY CHANGES IN BIOMARKERS\n")
        f.write("-" * 30 + "\n")
        for feature, change in results['top_changed_features'].items():
            f.write(f"{feature}: {change:.2f}% change in importance\n")
        
        f.write("\n")
        f.write("CONCLUSION\n")
        f.write("-" * 30 + "\n")
        if results['stress_reduction'] > 0:
            f.write(f"Yoga practice resulted in a {results['stress_reduction']:.2f}% reduction in stress levels.\n")
            if results['p_value'] < 0.05:
                f.write("This reduction is statistically significant, indicating that yoga has a genuine effect on reducing stress.\n")
            else:
                f.write("While a reduction was observed, it was not statistically significant, suggesting more data may be needed.\n")
        else:
            f.write("No stress reduction was observed. This may be due to experimental factors or data collection issues.\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nReport generated: {report_file}")

# Main function to call the analysis
if __name__ == "__main__":
    # Replace with your actual file paths
    before_yoga_file = "/Users/anchitmehra/Documents/coding/projects/Final_merged_dataset_with_weighted_stress.csv"
    after_yoga_file = "/Users/anchitmehra/Documents/coding/projects/synthetic_dataset.csv"
    
    # Run the analysis
    results = run_yoga_stress_analysis(before_yoga_file, after_yoga_file)