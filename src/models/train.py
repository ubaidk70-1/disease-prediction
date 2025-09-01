
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import os
import pickle

# --- File Paths ---
RAW_DATA_PATH = os.path.join('data', 'raw', 'Training.csv')
MODELS_DIR = os.path.join('models')

# --- Main Training Function ---
def train_final_model(raw_data_path):
    """
    Loads the full raw training data, preprocesses it, and trains the final
    LightGBM model with the best hyperparameters. Saves the trained model.
    """
    print("--- Starting Final Model Training ---")

    # --- Load the Raw Data ---
    try:
        df = pd.read_csv(raw_data_path)
        print("Raw training data loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: Raw data file not found at {raw_data_path}")
        return

    # --- Robust Data Cleaning ---
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    X = df.drop('prognosis', axis=1)
    y = df['prognosis']
    print(f"Training features prepared. Shape: {X.shape}")

    # --- Encode the Target Variable ---
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # --- Define Best Hyperparameters (from Optuna study) ---
    # These should be the exact values you found during experimentation
    best_params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 31,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'boosting_type': 'gbdt',
    }
    
    # --- Train the Final Model ---
    print("Training final LightGBM model on the full dataset...")
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X, y_encoded)
    print("✅ Final model trained successfully.")
    
    # --- Save the Trained Model ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'lgbm_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
    print(f"Final model saved to '{model_path}'")
    
    print("--- Final Model Training Complete ---")

# --- Main Execution Block ---
if __name__ == '__main__':
    train_final_model(RAW_DATA_PATH)
