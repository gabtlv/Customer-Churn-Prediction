# Male = 0 Female = 1
# Churn Yes = 1 No = 0
# Scaler is exported as scaler.pkl
# Model is exported as model.pkl
# Order of the X -> 'Age', 'Gender', 'Tenure', 'MonthlyCharges'

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Analytics", layout="wide")

#load assets
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")


st.title("Customer Churn Analytics")

#session state
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "probability" not in st.session_state:
    st.session_state.probability = None

predictionTab, dataTab = st.tabs(["Prediction Tool", "Data Insights"])

with predictionTab:
    st.subheader("Predict Customer Churn")
    toggle = st.toggle("Show Percentage")

    leftColumn, rightColumn = st.columns(2)

    with leftColumn:
        age = st.number_input("Enter age", min_value = 10, max_value = 100, value = 30)
        tenure = st.number_input("Enter Tenure (Months)", min_value = 0, max_value = 140)

    with rightColumn:
        monthlycharge = st.number_input("Enter Monthly Charges", min_value = 30, max_value = 140)
        gender = st.selectbox("Enter the gender",["Male","Female"])

    st.divider()

    predictButton = st.button("Run a Prediction")

    if predictButton:
        gender_selected = 1 if gender == "Female" else 0
        X = [age, gender_selected, tenure, monthlycharge]
        X1 = np.array(X)
        X_array = scaler.transform([X1])
        st.session_state.prediction = model.predict(X_array)[0]
        st.session_state.probability = model.predict_proba(X_array)[0][1]
        ##st.write(f"Raw Model Output: {model.predict_proba(X_array)}")   

    if st.session_state.prediction is not None:
        if toggle:
            st.write("---")
            metricColumn, progressColumn = st.columns([1,2])
            with metricColumn:
                st.metric("Churn Probability", f"{st.session_state.probability * 100:.1f}%")
            with progressColumn:
                st.write("Risk Confidence Level")
                st.progress(st.session_state.probability)
        else:
            if st.session_state.prediction == 1:
                st.error(f"### Result: High risk of Churn")
            else:
                st.success(f"### Result: Low risk of Churn")
    else:
        st.info("Please enter the values and run a prediction")

    with dataTab:
        st.subheader("Exploratory Data Analysis")
        SIZE = (5,5)
        @st.cache_data
        def load_data():
            return pd.read_csv("customer_churn_data.csv")
        
        df = load_data()

        column1, column2, column3 = st.columns(3)

        with column1:
            fig1, ax1 = plt.subplots(figsize=SIZE, constrained_layout=True)
            df["Churn"].value_counts().plot(kind="pie",autopct="%1.1f%%", startangle=90, title="Churn Yes or No", ylabel = "", ax=ax1)
            st.pyplot(fig1)
        
        with column2:
            fig2, ax2 = plt.subplots(figsize=SIZE, constrained_layout=True)
            df.groupby("ContractType")["MonthlyCharges"].mean().plot(kind="bar",ylabel="Monthly Charges",xlabel="Contract Type",
            title="Monthly Charges vs. Contract Type", ax=ax2)
            st.pyplot(fig2)
        
        with column3:
            fig3, ax3 = plt.subplots(figsize=SIZE, constrained_layout=True)
            df["Tenure"].plot(kind="hist",ylabel = "# of customers", xlabel = "Monthly Charges", title = "Histogram of Tenures", ax=ax3)
            st.pyplot(fig3)