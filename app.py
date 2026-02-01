import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("Heart Disease Prediction App")
st.write("Machine Learning models trained on Heart Disease dataset")

# Load models
models = {
    "Logistic Regression": joblib.load("model/Logistic_Regression.pkl"),
    "Decision Tree": joblib.load("model/Decision_Tree.pkl"),
    "KNN": joblib.load("model/KNN.pkl"),
    "Naive Bayes": joblib.load("model/Naive_Bayes.pkl"),
    "Random Forest": joblib.load("model/Random_Forest.pkl"),
    "XGBoost": joblib.load("model/XGBoost.pkl")
}

scaler = joblib.load("model/scaler.pkl")

# User input form
st.sidebar.header("Input Patient Details")

def user_input():
    age = st.sidebar.slider("Age", 20, 80, 40)
    sex = st.sidebar.selectbox("Sex", ["M", "F"])
    chest_pain = st.sidebar.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    resting_bp = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
    cholesterol = st.sidebar.slider("Cholesterol", 100, 600, 200)
    fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.sidebar.selectbox("Exercise Induced Angina", ["Y", "N"])
    oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)
    st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

    data = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input()

st.subheader("Input Data")
st.write(input_df)

# Encode categorical variables
input_df_encoded = pd.get_dummies(input_df)
input_df_encoded = input_df_encoded.reindex(columns=scaler.feature_names_in_, fill_value=0)

# Scale
input_scaled = scaler.transform(input_df_encoded)

# Model selection
model_name = st.selectbox("Choose Model", list(models.keys()))
model = models[model_name]

# Prediction
prediction = model.predict(input_scaled)[0]
prediction_proba = model.predict_proba(input_scaled)[0][1]

st.subheader("Prediction Result")
if prediction == 1:
    st.error(f"⚠️ Heart Disease Detected (Probability: {prediction_proba:.2f})")
else:
    st.success(f"✅ No Heart Disease Detected (Probability: {prediction_proba:.2f})")
