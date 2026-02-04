# Male = 0 Female = 1
# Churn Yes = 1 No = 0
# Scaler is exported as scaler.pkl
# Model is exported as model.pkl
# Order of the X -> 'Age', 'Gender', 'Tenure', 'MonthlyCharges'

import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

#session state
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "probability" not in st.session_state:
    st.session_state.probability = None

st.title("Churn Prediction App")
st.divider()
st.write("Please enter the values and hit the predict button to get a prediction.")
st.divider()

toggle = st.toggle("Show Percentage")
age = st.number_input("Enter age", min_value = 10, max_value = 100, value = 30)
tenure = st.number_input("Enter Tenure (Months)", min_value = 0, max_value = 140)
monthlycharge = st.number_input("Enter Monthly Charges", min_value = 30, max_value = 140)
gender = st.selectbox("Enter the gender",["Male","Female"])

st.divider()

predictButton = st.button("Predict!")

if predictButton:
    gender_selected = 1 if gender == "Female" else 0
    X = [age, gender_selected, tenure, monthlycharge]
    X1 = np.array(X)
    X_array = scaler.transform([X1])
    st.session_state.prediction = model.predict(X_array)[0]
    st.session_state.probability = model.predict_proba(X_array)[0][1]

if st.session_state.prediction is not None:
    if toggle:
        st.metric("Churn Probability", f"{st.session_state.probability * 100:.1f}%")
    else:
        result = ("Churn" if st.session_state.prediction == 1 else "Not Churn")
        st.write(f"Prediction: {result}")
else:
    st.write("Please enter the values and use predict button")