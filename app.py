import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Heart Disease Classification App", layout="wide")

st.title("❤️ Heart Disease Classification App")
st.write("Upload a CSV file (heart.csv) and evaluate multiple ML classification models")

# -------------------- FILE UPLOAD --------------------
uploaded_file = st.file_uploader(
    "Upload Heart Disease CSV file (Test Data Only)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# -------------------- LOAD DATA --------------------
df = pd.read_csv(uploaded_file)
st.subheader("📄 Uploaded Dataset Preview")
st.dataframe(df.head())

# -------------------- TARGET & FEATURES --------------------
TARGET_COL = "HeartDisease"

if TARGET_COL not in df.columns:
    st.error("Target column 'HeartDisease' not found in dataset.")
    st.stop()

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# -------------------- ENCODING --------------------
categorical_cols = X.select_dtypes(include="object").columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# -------------------- TRAIN TEST SPLIT --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------- FEATURE SCALING --------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------- MODEL SELECTION --------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
}

st.subheader("🤖 Select Machine Learning Model")
model_name = st.selectbox("Choose a model", list(models.keys()))
model = models[model_name]

# -------------------- TRAIN MODEL --------------------
model.fit(X_train, y_train)

# -------------------- PREDICTION --------------------
y_pred = model.predict(X_test)

if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_test)[:, 1]
else:
    y_prob = y_pred

# -------------------- METRICS --------------------
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

# -------------------- DISPLAY METRICS --------------------
st.subheader("📊 Evaluation Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy:.3f}")
col2.metric("AUC", f"{auc:.3f}")
col3.metric("Precision", f"{precision:.3f}")

col4, col5, col6 = st.columns(3)
col4.metric("Recall", f"{recall:.3f}")
col5.metric("F1 Score", f"{f1:.3f}")
col6.metric("MCC", f"{mcc:.3f}")

# -------------------- CONFUSION MATRIX --------------------
st.subheader("🧩 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# -------------------- CLASSIFICATION REPORT --------------------
st.subheader("📑 Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

st.subheader("Prediction Result")
if prediction == 1:
    st.error(f"⚠️ Heart Disease Detected (Probability: {prediction_proba:.2f})")
else:
    st.success(f"✅ No Heart Disease Detected (Probability: {prediction_proba:.2f})")
