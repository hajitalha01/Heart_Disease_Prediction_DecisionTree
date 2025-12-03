import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open('HeartDisease_Model.pkl', 'rb') as f:
    model = pickle.load(f)

# App title
st.title("Heart Disease Prediction App")
st.write("Predict whether a patient has heart disease using a trained Decision Tree model.")

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
        'RestingE
