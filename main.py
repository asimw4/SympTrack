import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import json

# Load trained model
model = joblib.load("risk_model.pkl")


# Set Streamlit config
st.set_page_config(page_title="SympTrack", page_icon="🫀", layout="centered")

# App Title
st.title("🫀 SympTrack: Heart Health Insight Tool")
st.subheader("Enter your health info to get an AI-powered risk insight")

st.sidebar.title("ℹ️ About SympTrack")
st.sidebar.markdown("""
**Model:** Logistic Regression  
**Accuracy:** 88%  
**F1 Score:** 86%  

**Dataset:** UCI Heart Disease (Cleveland subset)  

This tool is for **educational purposes only** and does not provide medical advice.

**Model quality:**  
- **Accuracy** = % of total cases predicted correctly.  
- **F1 Score** = Balances precision and recall — reflects ability to detect heart disease cases while minimizing false positives.

**Explainability:** Feature contributions shown below.
""")

# Definitions for key terms
definitions = {
    'cp': "Chest Pain Type — pain experienced in the chest, often related to heart stress.",
    'thalach': "Maximum Heart Rate Achieved during physical activity.",
    'oldpeak': "ST depression during exercise compared to rest (indicator of ischemia).",
    'trestbps': "Resting Blood Pressure — the blood pressure when you’re at rest.",
    'chol': "Serum Cholesterol — amount of cholesterol in mg/dl.",
    'thal': "Thalassemia — a blood disorder affecting oxygen transport.",
    'slope': "Slope of the ST segment in an ECG (Upsloping, Flat, or Downsloping).",
    'fbs': "Fasting Blood Sugar > 120 mg/dl (1 = yes).",
    'exang': "Exercise-induced angina (chest pain).",
    'ca': "Number of major vessels colored by fluoroscopy (0–3)."
}

# User inputs
age = st.slider("Age", 20, 80, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", ["Typical Angina (0)", "Atypical Angina (1)", "Non-anginal Pain (2)", "Asymptomatic (3)"])
trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
chol = st.slider("Cholesterol (mg/dl)", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", ["No (0)", "Yes (1)"])
restecg = st.selectbox("Resting ECG Result (restecg)", ["Normal (0)", "ST-T Abnormality (1)", "Left Ventricular Hypertrophy (2)"])
thalach = st.slider("Max Heart Rate Achieved (thalach)", 70, 210, 150)
exang = st.selectbox("Exercise-Induced Angina (exang)", ["No (0)", "Yes (1)"])
oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", ["Upsloping (0)", "Flat (1)", "Downsloping (2)"])
ca = st.slider("Number of Major Vessels Colored (ca)", 0, 3, 0)
thal = st.selectbox("Thalassemia Type (thal)", ["Normal (3)", "Fixed Defect (6)", "Reversible Defect (7)"])

# Convert inputs to numeric
sex = 1 if sex == "Male" else 0
cp = int(cp.split("(")[-1][0])
fbs = int(fbs.split("(")[-1][0])
restecg = int(restecg.split("(")[-1][0])
exang = int(exang.split("(")[-1][0])
slope = int(slope.split("(")[-1][0])
thal = int(thal.split("(")[-1][0])

# Create user input array
user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

# When button is pressed
if st.button("Run Health Insight"):
    proba = model.predict_proba(user_input)[0][1]
    prediction = int(proba >= 0.5)
    risk_percent = round(proba * 100)

    st.markdown(f"🧠 **Risk Similarity Score:** {risk_percent}%")

    # Feature importance (manual)
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    coefs = model.coef_[0]
    impacts = []
    for name, coef, val in zip(feature_names, coefs, user_input[0]):
        impact_value = round(abs(coef * val), 2)
        sign = np.sign(coef * val)
        direction = "↑ increases risk" if sign > 0 else "↓ decreases risk"
        impacts.append((name, impact_value, direction))

    # Sort impacts
    top_features = sorted(impacts, key=lambda x: x[1], reverse=True)[:3]

    # Display Top Factors
    st.markdown("### 🔍 Top Factors Influencing This Result:")
    for name, impact_value, direction in top_features:
        st.write(f"- **{name.capitalize()}** (Impact Score: {impact_value}) — *{direction}*")
        st.caption(definitions.get(name, "No description available."))


    # Full feature impact bar chart
    st.markdown("### 📊 Full Model Feature Impact")
    all_features_sorted = sorted(impacts, key=lambda x: x[1], reverse=True)
    feature_names_plot = [name.capitalize() for name, impact, direction in all_features_sorted]
    feature_impacts_plot = [impact for name, impact, direction in all_features_sorted]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(feature_names_plot, feature_impacts_plot, color="#2e86de")
    ax.invert_yaxis()
    ax.set_xlabel("Impact Score")
    ax.set_title("Feature Contributions to Current Prediction")

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}', va='center')

    st.pyplot(fig)

    # Risk assessment result
    if prediction == 1:
        st.error("⚠️ Your health profile is similar to individuals with heart conditions in the dataset. Consider speaking to a healthcare provider.")
    else:
        st.success("✅ Your inputs do not strongly match patterns seen in heart disease cases in the dataset.")

# Educational info
with st.expander("🧠 How this works"):
    st.markdown("""
    We use a machine learning model trained on real-world heart disease data.  
    Your inputs are analyzed to estimate how similar they are to profiles of individuals with heart disease in the dataset.  
    This result is not a diagnosis, but a data-driven insight into risk pattern similarity.

    **Common Terms:**
    - **Thalach**: Max heart rate during exercise  
    - **Oldpeak**: ST depression from ECG  
    - **Cp**: Chest pain type  
    """)
