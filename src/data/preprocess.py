# src/data/preprocess.py

# --- Import necessary libraries ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import pickle

# --- File Paths ---
# Define paths relative to the project's root directory
RAW_DATA_PATH = os.path.join('data', 'raw', 'Training.csv')
PROCESSED_DATA_DIR = os.path.join('data', 'processed')
MODELS_DIR = os.path.join('models')

# --- Main Preprocessing Function ---
def preprocess_data(raw_data_path):
    """
    Loads, cleans, and splits the raw training data.
    Saves the processed data and the label encoder.
    """
    print("--- Starting Data Preprocessing ---")
    
    # --- Load the Raw Data ---
    try:
        df = pd.read_csv(raw_data_path)
        print(f"Raw data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f" Error: Raw data file not found at {raw_data_path}")
        return

    # --- Robust Data Cleaning ---
    # 1. Clean up column names by stripping whitespace
    df.columns = df.columns.str.strip()
    
    # 2. Drop any unnamed columns that may have been created from trailing commas
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # 3. Explicitly drop the target column 'prognosis' to create features (X)
    X = df.drop('prognosis', axis=1)
    
    # 4. Create the target variable (y)
    y = df['prognosis']
    
    print(f"Features (X) created successfully. Shape: {X.shape}")
    print(f"Target (y) created successfully. Shape: {y.shape}")

    # --- Encode the Target Variable ---
    # Machine learning models require numerical inputs, so we convert text labels to numbers.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print("Target variable 'prognosis' encoded successfully.")

    # --- Split Data into Training and Validation Sets ---
    # We split the data to train the model on one part and evaluate it on another.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, 
        test_size=0.2,    # 20% of the data will be used for validation
        random_state=42,  # Ensures the split is the same every time
        stratify=y_encoded # Ensures the proportion of diseases is the same in train and val sets
    )
    print("Data split into training and validation sets.")

    # --- Save Processed Data and Encoder ---
    # Ensure the destination directories exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save the data splits as CSV files
    pd.DataFrame(X_train).to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'), index=False)
    pd.DataFrame(X_val).to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_val.csv'), index=False)
    pd.DataFrame(y_train, columns=['prognosis']).to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'), index=False)
    pd.DataFrame(y_val, columns=['prognosis']).to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_val.csv'), index=False)
    print(f"Processed data saved to '{PROCESSED_DATA_DIR}'")

    # Save the label encoder object using pickle for later use in predictions
    with open(os.path.join(MODELS_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved to '{MODELS_DIR}'")
    
    print("--- Data Preprocessing Complete ---")


# --- Main Execution Block ---
if __name__ == '__main__':
    # This block runs only when the script is executed directly
    preprocess_data(RAW_DATA_PATH)
