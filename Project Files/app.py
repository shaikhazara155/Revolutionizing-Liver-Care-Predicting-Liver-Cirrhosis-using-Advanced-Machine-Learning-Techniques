import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model and preprocessing tools
with open("best_model.pkl", "rb") as file:
    model, scaler, label_encoders, feature_names = pickle.load(file)

st.set_page_config(page_title="Liver Cirrhosis Detection", page_icon="🩺")
st.title("🩺 Liver Cirrhosis Prediction")
st.markdown("Enter the patient details below:")

# Normalize encoder keys
normalized_encoders = {key.lower(): val for key, val in label_encoders.items()}

# Collect user input
user_input = {}
for feature in feature_names:
    feature_lower = feature.lower()
    if feature_lower in normalized_encoders:
        le = normalized_encoders[feature_lower]
        options = list(le.classes_)
        choice = st.selectbox(f"{feature}", options)
        user_input[feature] = le.transform([choice])[0]
    elif "age" in feature_lower:
        user_input[feature] = st.number_input(f"{feature}", min_value=1, max_value=120, step=1)
    else:
        user_input[feature] = st.number_input(f"{feature}", format="%.2f")

# Predict user input
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    input_df = input_df[feature_names]  # Ensure correct order
    input_scaled = scaler.transform(input_df)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_scaled)[0][1]
        prediction = 1 if prob > 0.4 else 0
    else:
        prediction = model.predict(input_scaled)[0]
        prob = None

    result = "🛑 Has Liver Cirrhosis" if prediction == 1 else "✅ Healthy"
    st.success(f"Prediction Result: {result}")
    if prob is not None:
        st.info(f"Model Confidence (Cirrhosis Probability): {prob * 100:.2f}%")

# Auto test with real cirrhosis patient
st.markdown("---")
st.subheader("📊 Auto Test: Real Cirrhosis Patient From Excel")

try:
    df = pd.read_excel("HealthCareData.xlsx")
    df.columns = df.columns.str.strip()
    df.drop(columns=['S.NO'], errors='ignore', inplace=True)

    target_col = "Predicted Value(Out Come-Patient suffering from liver  cirrosis or not)"
    df[target_col] = df[target_col].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    real_case = df[df[target_col] == 1].iloc[0:1].copy()
    true_label = real_case[target_col].values[0]
    real_case.drop(columns=[target_col], inplace=True)

    for col in real_case.columns:
        if col in label_encoders:
            le = label_encoders[col]
            real_case[col] = le.transform(real_case[col].astype(str))

    real_case = real_case[feature_names]
    real_scaled = scaler.transform(real_case)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(real_scaled)[0][1]
        prediction = 1 if prob > 0.4 else 0
    else:
        prediction = model.predict(real_scaled)[0]
        prob = None

    result = "🛑 Has Liver Cirrhosis" if prediction == 1 else "✅ Healthy"
    st.write("✅ **Auto-Test Result on Real Cirrhosis Case**")
    st.write(f"🔍 Prediction: {result}")
    if prob is not None:
        st.write(f"📈 Model Confidence: **{prob * 100:.2f}%**")
    st.write(f"📌 Actual Label in Excel: {'🛑 Cirrhosis' if true_label == 1 else '✅ Healthy'}")

except Exception as e:
    st.error(f"⚠️ Error during test: {e}")
