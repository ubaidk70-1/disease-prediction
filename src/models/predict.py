import pickle
import os
import numpy as np
import pandas as pd

# --- Define paths to our model and encoder ---
# We assume this script is run from the project root, so paths are relative to that.
MODEL_PATH = os.path.join('models', 'lgbm_model.pkl')
ENCODER_PATH = os.path.join('models', 'label_encoder.pkl')

# --- Load the model and encoder at script startup ---
# This is efficient because we don't reload the model every time we make a prediction.
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(" Model loaded successfully.")
except FileNotFoundError:
    print(f" Error: Model not found at {MODEL_PATH}. Please run the training script first.")
    model = None

try:
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print(" Label encoder loaded successfully.")
except FileNotFoundError:
    print(f" Error: Encoder not found at {ENCODER_PATH}. Please run the preprocessing script first.")
    label_encoder = None


def make_prediction(symptoms_array):
    """
    Makes a disease prediction based on a numpy array of symptoms.

    Args:
        symptoms_array (np.array): A 1D numpy array of shape (132,)
                                   representing the patient's symptoms (0s and 1s).

    Returns:
        str: The predicted disease name. Returns an error message if models are not loaded.
    """
    # Ensure our models are loaded before trying to predict
    if model is None or label_encoder is None:
        return "Error: Model or encoder not loaded. Cannot make a prediction."

    # The model expects a 2D array for prediction, so we reshape our 1D array
    # from (132,) to (1, 132)
    symptoms_reshaped = symptoms_array.reshape(1, -1)

    # Use the model to predict the encoded label (e.g., 5)
    prediction_encoded = model.predict(symptoms_reshaped)

    # Use the encoder to transform the encoded label back to the original disease name
    # (e.g., 5 -> 'Jaundice')
    prediction_disease = label_encoder.inverse_transform(prediction_encoded)

    # The result is an array with one item, so we return the first item
    return prediction_disease[0]


if __name__ == '__main__':
    # This block is for testing the function directly.
    # It will only run when you execute 'python src/models/predict.py'
    print("\n--- Running a test prediction ---")

    # Let's create a test case based on the Jaundice symptoms from our EDA
    # First, get all the column names from the training data to ensure correct order
    try:
        symptom_columns = pd.read_csv('data/processed/X_train.csv').columns
        test_symptoms = pd.Series(0, index=symptom_columns) # Create a series of all zeros

        # Set the symptoms for Jaundice to 1
        jaundice_symptoms = ['itching', 'weight_loss', 'fatigue', 'vomiting',
                             'dark_urine', 'yellowish_skin', 'abdominal_pain', 'high_fever']
        test_symptoms.loc[jaundice_symptoms] = 1

        # Convert the pandas Series to a numpy array for our function
        test_symptoms_array = test_symptoms.to_numpy()

        # Make the prediction
        predicted_disease = make_prediction(test_symptoms_array)
        print(f"Test Symptoms: Jaundice-like")
        print(f"Predicted Disease: '{predicted_disease}'")
        print(f"Test successful: {'expected prediction' if predicted_disease == 'Jaundice' else '(Unexpected prediction)'}")

    except FileNotFoundError:
        print(" Could not run test: 'data/processed/X_train.csv' not found.")
