# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ===========================
# Load Models and Objects
# ===========================

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

churn_model = joblib.load(os.path.join(MODELS_DIR, "churn_model.pkl"))
engagement_model = joblib.load(os.path.join(MODELS_DIR, "engagement_model.pkl"))
clv_model = joblib.load(os.path.join(MODELS_DIR, "clv_model.pkl"))

scaler_churn = joblib.load(os.path.join(MODELS_DIR, "scaler_churn.pkl"))
scaler_engagement = joblib.load(os.path.join(MODELS_DIR, "scaler_engagement.pkl"))
scaler_clv = joblib.load(os.path.join(MODELS_DIR, "scaler_clv.pkl"))

feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))

# ===========================
# Helper Functions
# ===========================

def preprocess_input(input_df):
    cat_cols = [
        "gender", "subscription_type", "region",
        "device", "payment_method", "favorite_genre"
    ]

    input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

    # Align exactly with training features
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    return input_encoded


def classify_churn_risk(prob):
    if prob < 0.40:
        return "Low Risk", "Customer is stable. Focus on loyalty."
    elif prob <= 0.70:
        return "Medium Risk", "Customer requires engagement push."
    else:
        return "High Risk", "Immediate retention action needed."


# ===========================
# Streamlit UI
# ===========================

st.set_page_config(page_title="Netflix Customer Prediction Tool", layout="centered")
st.title("Netflix Customer Status Prediction")
st.write("Enter customer details to predict churn risk, engagement level, and CLV tier.")

with st.form("customer_form"):
    st.header("Enter Customer Details")

    # Numerical Inputs
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    monthly_fee = st.number_input("Monthly Fee", min_value=0.0, value=13.99, format="%.2f")
    watch_hours = st.number_input("Watch Hours", min_value=0.0, value=10.0, format="%.2f")
    last_login_days = st.number_input("Last Login Days", min_value=0, value=5)
    number_of_profiles = st.number_input("Number of Profiles", min_value=1, value=1)
    avg_watch_time_per_day = st.number_input(
        "Average Watch Time Per Day", min_value=0.0, value=1.0, format="%.2f"
    )

    # Dropdown Inputs (MATCH TRAINING DATA)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    region = st.selectbox(
        "Region",
        ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]
    )
    device = st.selectbox("Device", ["Laptop", "Mobile", "TV", "Tablet"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Credit Card", "Debit Card", "PayPal", "Crypto", "Gift Card"]
    )
    favorite_genre = st.selectbox(
        "Favorite Genre",
        ["Action", "Comedy", "Documentary", "Drama", "Horror", "Romance", "Sci-Fi"]
    )

    submitted = st.form_submit_button("Predict Customer Status")


# ===========================
# Prediction Logic
# ===========================

if submitted:
    # Prepare input dataframe
    input_dict = {
        "age": [age],
        "monthly_fee": [monthly_fee],
        "watch_hours": [watch_hours],
        "last_login_days": [last_login_days],
        "number_of_profiles": [number_of_profiles],
        "avg_watch_time_per_day": [avg_watch_time_per_day],
        "gender": [gender],
        "subscription_type": [subscription_type],
        "region": [region],
        "device": [device],
        "payment_method": [payment_method],
        "favorite_genre": [favorite_genre],
    }

    input_df = pd.DataFrame(input_dict)

    # Feature engineering
    input_df["total_charges"] = input_df["watch_hours"] * input_df["monthly_fee"]

    # Preprocess input
    processed_df = preprocess_input(input_df)

    # Apply scalers
    X_churn = scaler_churn.transform(processed_df)
    X_engagement = scaler_engagement.transform(processed_df)
    X_clv = scaler_clv.transform(processed_df)

    # Predictions
    churn_pred = churn_model.predict(X_churn)[0]
    churn_prob = churn_model.predict_proba(X_churn)[0][1]

    engagement_pred = engagement_model.predict(X_engagement)[0]
    clv_pred = clv_model.predict(X_clv)[0]

    # Churn risk classification
    churn_risk, churn_message = classify_churn_risk(churn_prob)

    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Churn Probability:** {churn_prob * 100:.2f}%")
    st.write(f"**Churn Risk Category:** {churn_risk}")
    st.write(f"**Engagement Level:** {engagement_pred}")
    st.write(f"**CLV Tier:** {clv_pred}")
    st.write(f"**Business Message:** {churn_message}")
