# src/app.py

# --- Import necessary libraries ---
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import os

# Important: We are importing our custom prediction function from the 'predict' module.
# This assumes you run the app from the root directory using 'python -m src.app'
from src.models.predict import make_prediction

# --- Initialize the Flask App ---
# We tell Flask where to find the template and static files relative to this script.
app = Flask(__name__, template_folder='../templates', static_folder='../static')


# --- Load Symptom List for the Form ---
# We need the list of all possible symptoms to display them as checkboxes in the HTML form.
try:
    # Construct the full path to the CSV file
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'Training.csv')
    symptom_data = pd.read_csv(csv_path)
    
    # --- ROBUST FIX ---
    # 1. Clean up column names by stripping whitespace
    symptom_data.columns = symptom_data.columns.str.strip()
    
    # 2. Drop any unnamed columns that may have been created from trailing commas
    symptom_data = symptom_data.loc[:, ~symptom_data.columns.str.contains('^Unnamed')]

    # 3. Drop the target column 'prognosis' explicitly by name.
    if 'prognosis' in symptom_data.columns:
        symptom_data = symptom_data.drop('prognosis', axis=1)
    
    # Get all column names for processing the form data
    SYMPTOM_COLUMNS = symptom_data.columns
    # Create a clean, sorted list for display on the webpage
    SYMPTOMS = sorted([col.replace('_', ' ').title() for col in SYMPTOM_COLUMNS])

    print(f" Symptom list loaded successfully with {len(SYMPTOM_COLUMNS)} features.")

except FileNotFoundError:
    print(f" Error: 'Training.csv' not found. Cannot start the app.")
    SYMPTOMS = []
    SYMPTOM_COLUMNS = []


# --- Define Routes ---

@app.route('/')
def home():
    """
    Renders the main page of the web application.
    This page contains the form where users can select their symptoms.
    """
    return render_template('index.html', symptoms=SYMPTOMS)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the user.
    """
    # Get the list of symptoms checked by the user from the form
    selected_symptoms_display = request.form.getlist('symptoms')
    print(f"Received symptoms from user: {selected_symptoms_display}")

    # --- Create the Input Array for the Model ---
    # Start with an array of all zeros, one for each possible symptom
    symptoms_array = np.zeros(len(SYMPTOM_COLUMNS))
    
    # Convert the user-friendly display names back to the original column format
    selected_symptoms_original = [s.lower().replace(' ', '_') for s in selected_symptoms_display]
    
    # Go through our master list of symptoms and set the value to 1 if the user selected it
    for i, col in enumerate(SYMPTOM_COLUMNS):
        if col in selected_symptoms_original:
            symptoms_array[i] = 1

    # --- Make the Prediction ---
    predicted_disease = make_prediction(symptoms_array)
    print(f"Model prediction: {predicted_disease}")

    # --- Display the Result ---
    # Send the predicted disease name to the result.html template
    return render_template('result.html', prediction=predicted_disease)


# --- Run the App ---
if __name__ == '__main__':
    # This block allows the app to be run directly with 'python src/app.py'
    # or with 'python -m src.app'
    app.run(debug=True)
