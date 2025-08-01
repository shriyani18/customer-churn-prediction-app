import streamlit as st
import pandas as pd
import pickle

# Load model, features, and encoders once when app starts
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
loaded_model = model_data["model"]
feature_names = model_data["features_names"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# UI Inputs
st.title("Customer Churn Prediction")

gender = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=119.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, value=8685.0)

# On button click, run prediction
if st.button("Predict Churn"):
    input_data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    for col, encoder in encoders.items():
        input_df[col] = encoder.transform(input_df[col])

    # Reorder columns to match training
    input_df = input_df[feature_names]

    # Predict
    prediction = loaded_model.predict(input_df)
    pred_prob = loaded_model.predict_proba(input_df)

    churn_status = "Churn" if prediction[0] == 1 else "No Churn"
    churn_prob = pred_prob[0][1]

    st.write(f"### Prediction: {churn_status}")
    st.write(f"### Probability of churn: {churn_prob:.2f}")
