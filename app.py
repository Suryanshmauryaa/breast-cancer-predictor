"""
Streamlit demo for the Breast Cancer Diagnosis Predictor.

Run:
    streamlit run app.py
"""

import json
import joblib
import numpy as np
import streamlit as st
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="Breast Cancer Diagnosis Predictor", layout="centered")

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

with open("models/results.json") as f:
    results = json.load(f)

data = load_breast_cancer()

st.title("🩺 Breast Cancer Diagnosis Predictor")
st.write(
    "Predicts whether a tumor is **benign** or **malignant** from 30 diagnostic "
    "features, using a model trained on the Wisconsin Breast Cancer dataset."
)

st.info(f"Model in use: **{results['best_model']}** "
        f"(F1-score: {results['all_results'][results['best_model']]['f1_score']}, "
        f"ROC-AUC: {results['all_results'][results['best_model']]['roc_auc']})")

st.subheader("Try it out")
st.caption("Load a random real sample or adjust values manually.")

if "sample_idx" not in st.session_state:
    st.session_state.sample_idx = 0

if st.button("🎲 Load random sample from dataset"):
    st.session_state.sample_idx = np.random.randint(0, len(data.data))

sample = data.data[st.session_state.sample_idx]
actual_label = data.target_names[data.target[st.session_state.sample_idx]]

inputs = []
cols = st.columns(3)
for i, fname in enumerate(feature_names):
    col = cols[i % 3]
    val = col.number_input(fname, value=float(sample[i]), format="%.4f")
    inputs.append(val)

st.caption(f"Actual label for loaded sample: **{actual_label}**")

if st.button("Predict"):
    X = scaler.transform([inputs])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    label = "Benign" if pred == 1 else "Malignant"
    confidence = proba[pred]

    if pred == 1:
        st.success(f"Prediction: **{label}** (confidence: {confidence:.2%})")
    else:
        st.error(f"Prediction: **{label}** (confidence: {confidence:.2%})")

st.divider()
st.subheader("Model comparison")
st.json(results["all_results"])
