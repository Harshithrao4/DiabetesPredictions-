import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load Dataset
data = pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\dataset.csv")

# Preprocessing
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# Streamlit UI
st.set_page_config(page_title="Diabetes Prediction System", layout="wide")

st.markdown("<h1 style='text-align: center; color: white;'>WELCOME TO DIABETES PREDICTION SYSTEM</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'><a href='#predict'><button style='background-color: lightblue; padding: 10px 20px; border-radius: 10px; font-size: 20px;'>Let's Get Started!</button></a></p>", unsafe_allow_html=True)

st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/EKG.svg/1920px-EKG.svg.png", use_column_width=True)

st.markdown("<h2 id='predict' style='text-align: center; color: lightblue;'>Enter Your Information Below:</h2>", unsafe_allow_html=True)

Pregnancies = st.number_input("Pregnancies", min_value=0)
Glucose = st.number_input("Glucose Level", min_value=0)
BloodPressure = st.number_input("Blood Pressure", min_value=0)
SkinThickness = st.number_input("Skin Thickness", min_value=0)
Insulin = st.number_input("Insulin Level", min_value=0)
BMI = st.number_input("BMI", min_value=0.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0)
Age = st.number_input("Age", min_value=0)

if st.button("Submit"):
    user_input = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)

    if prediction[0] == 1:
        st.error("Result: You are Diabetic.")
    else:
        st.success("Result: You are NOT Diabetic.")
