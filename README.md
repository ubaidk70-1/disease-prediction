# Intelligent Health Predictor: AI-Powered Disease Prediction

An end-to-end machine learning application built for the **PWSkills Mini-Hackathon**.  
This project uses a dataset of patient symptoms to predict one of **41 possible diseases**, achieving **97.62% accuracy** on unseen test data. The solution is deployed as a user-friendly and interactive web application using **Flask**.

---

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Click%20Here-brightgreen?style=for-the-badge)](https://ubaidkha-disease-predictor-app.hf.space)

---

##  Application Interface

<img width="1522" height="881" alt="image" src="https://github.com/user-attachments/assets/71e558e0-039b-47af-a2d6-6b6b115cf0f7" />


---

##  Key Features
- **High-Accuracy Predictions**: Tuned LightGBM model with **97.62% accuracy** on the test set.  
- **Interactive UI**: Modern, responsive interface with a searchable list of **132 symptoms**.  
- **Dynamic Feedback**: Real-time updates as users select/deselect symptoms.  
- **End-to-End Pipeline**: Modular pipeline for preprocessing, training, evaluation, and prediction.  
- **Reproducible Environment**: Fully containerized with `requirements.txt` for easy setup.  
- **Thorough Analysis**: Includes detailed **EDA** to uncover insights from the dataset.  

---

##  Tech Stack
- **Language**: Python 3.9  
- **Libraries**: Pandas, NumPy, Scikit-learn, LightGBM, Optuna (hyperparameter tuning)  
- **Web Framework**: Flask  
- **Front-End**: HTML, CSS, JavaScript  
- **Development**: Jupyter Notebooks, Git  
- **Deployment**: Hugging Face Spaces  

---

##  Project Structure
```
disease-prediction-hackathon/
│
├── Notebooks/              # Jupyter notebooks for EDA and experimentation
│   ├── exploratory_data_analysis.ipynb
│   ├── data_preparation.ipynb
│   └── model_training.ipynb
│
├── src/                    # Source code for the application
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── __init__.py
│   ├── app.py
│   └── evaluate.py
│
├── data/
│   ├── raw/
│   │   ├── Training.csv
│   │   └── Testing.csv
│   └── processed/
│
├── models/       # Saved model (.pkl) and label encoder
│       ├── lgbm_model.pkl
│       └── label_encoder.pkl        
│
├── static/
│   └── style.css
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── predictions.csv
├── requirements.txt
└── README.md

```

## Setup and Installation (Local)

### 1. Clone the Repository
```
git clone https://github.com/<your-username>/disease-prediction-hackathon.git
cd disease-prediction-hackathon
```
### 2. Create and Activate a Virtual Environment
```
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

## How to Run Locally

### 1. Preprocess the Data

```
python src/data/preprocess.py
```
### 2. Train the Final Model
```
python src/models/train.py
```

### 3. Launch the Web Application
```
python -m src.app
```

Then, open your browser and go to:
http://127.0.0.1:5000

## Model Evaluation

To evaluate the model on the test set and generate the predictions.csv file:
```
python src/evaluate.py
```

This will print the final accuracy score in the console.
