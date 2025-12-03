import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open('HeartDisease_Model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Heart Disease Prediction App")
st.write("Predict whether a patient has heart disease using Decision Tree model.")

# Sidebar for user input
st.sidebar.header("Enter Patient Details")

def user_input_features():
    Age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
    Sex = st.sidebar.selectbox("Sex", ["M", "F"])
    ChestPainType = st.sidebar.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
    RestingBP = st.sidebar.number_input("Resting BP (mmHg)", min_value=50, max_value=250, value=120)
    Cholesterol = st.sidebar.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    FastingBS = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    RestingECG = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    MaxHR = st.sidebar.number_input("Max Heart Rate", min_value=60, max_value=250, value=150)
    ExerciseAngina = st.sidebar.selectbox("Exercise Induced Angina", ["Y", "N"])
    Oldpeak = st.sidebar.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    ST_Slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

    data = {
        'Age': Age,
        'RestingBP': RestingBP,
        'Cholesterol': Cholesterol,
        'FastingBS': FastingBS,
        'MaxHR': MaxHR,
        'Oldpeak': Oldpeak,
        'Sex': Sex,
        'ChestPainType': ChestPainType,
        'RestingECG': RestingECG,
        'ExerciseAngina': ExerciseAngina,
        'ST_Slope': ST_Slope
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# One-hot encode to match training
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

# Align input with model features
model_features = model.feature_names_in_
input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

# Prediction
prediction = model.predict(input_encoded)
prediction_proba = model.predict_proba(input_encoded)

st.subheader("Prediction")
heart_status = 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'
st.write(heart_status)

st.subheader("Prediction Probability")
st.write(f"No Heart Disease: {prediction_proba[0][0]:.2f}")
st.write(f"Heart Disease: {prediction_proba[0][1]:.2f}")

# Download button for .pkl file
with open('heart_disease_decision_tree.pkl', 'rb') as f:
    model_bytes = f.read()

st.download_button(
    label="Download Trained Model",
    data=model_bytes,
    file_name="HeartDisease_Model.pkl",
    mime="application/octet-stream"
)
