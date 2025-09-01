import pandas as pd
from sklearn.metrics import accuracy_score
import os
import pickle

# --- File Paths ---
TEST_DATA_PATH = os.path.join('data', 'raw', 'Testing.csv')
MODEL_PATH = os.path.join('models', 'lgbm_model.pkl')
ENCODER_PATH = os.path.join('models', 'label_encoder.pkl')
PREDICTIONS_PATH = 'predictions.csv'

# --- Main Evaluation Function ---
def evaluate_model(test_data_path):
    """
    Loads the unseen test data, makes predictions using the trained model,
    calculates the final accuracy, and saves the predictions to a CSV file.
    """
    print("--- Starting Final Model Evaluation on Test Data ---")
    
    # --- Load Model and Encoder ---
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)
        print(" Model and label encoder loaded successfully.")
    except FileNotFoundError:
        print(f" Error: Model or encoder not found. Please run train.py first.")
        return

    # --- Load and Process Test Data ---
    try:
        df_test = pd.read_csv(test_data_path)
        print("Test data loaded successfully.")
    except FileNotFoundError:
        print(f" Error: Test data file not found at {test_data_path}")
        return
        
    # --- Robust Data Cleaning ---
    df_test.columns = df_test.columns.str.strip()
    df_test = df_test.loc[:, ~df_test.columns.str.contains('^Unnamed')]
    
    # Separate features and true labels
    X_test = df_test.drop('prognosis', axis=1)
    y_test_true_labels = df_test['prognosis']
    y_test_encoded = encoder.transform(y_test_true_labels)
    
    print(f"Test features prepared. Shape: {X_test.shape}")

    # --- Make Predictions ---
    predictions_encoded = model.predict(X_test)
    
    # --- Calculate Final Accuracy ---
    accuracy = accuracy_score(y_test_encoded, predictions_encoded)
    print(f" Final Model Accuracy on Unseen Test Data: {accuracy:.2%}")
    
    # --- Save Predictions ---
    # Decode predictions back to original disease names
    predictions_decoded = encoder.inverse_transform(predictions_encoded)
    
    # Create a DataFrame to save
    predictions_df = pd.DataFrame({
        'Actual_Prognosis': y_test_true_labels,
        'Predicted_Prognosis': predictions_decoded
    })
    
    # Save to CSV
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f" Predictions saved to '{PREDICTIONS_PATH}'")
    
    print("--- Final Model Evaluation Complete ---")

# --- Main Execution Block ---
if __name__ == '__main__':
    evaluate_model(TEST_DATA_PATH)
