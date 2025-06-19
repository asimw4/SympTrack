import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load model + columns
diabetes_model = joblib.load("diabetes_model.pkl")
diabetes_columns = joblib.load("diabetes_columns.pkl")

def render():
    st.markdown("### 🍩 Diabetes Risk")

    # Input fields
    pregnancies = st.slider("Pregnancies", 0, 20, 1)
    glucose = st.slider("Glucose", 50, 200, 100)
    bp = st.slider("Blood Pressure", 40, 120, 70)
    skin = st.slider("Skin Thickness", 0, 100, 20)
    insulin = st.slider("Insulin", 0, 850, 80)
    bmi = st.slider("BMI", 10.0, 60.0, 30.0)
    pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.slider("Age", 10, 100, 35)

    # Create input frame
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]])
    df_input = pd.DataFrame(input_data, columns=diabetes_columns)

    # Predict
    proba = diabetes_model.predict_proba(df_input)[0][1]
    prediction = int(proba >= 0.5)
    risk_percent = round(proba * 100)

    st.markdown(f"🧠 **Risk Similarity Score:** {risk_percent}%")

    # Feature importance (Random Forest)
    importances = diabetes_model.feature_importances_
    feature_importance = list(zip(diabetes_columns, importances))
    all_features_sorted = sorted(feature_importance, key=lambda x: x[1], reverse=True)

    # Plot
    st.markdown("### 📊 Full Model Feature Impact")
    names = [name for name, _ in all_features_sorted]
    values = [val for _, val in all_features_sorted]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, values, color="#f39c12")
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Contributions to Diabetes Prediction")

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{width:.2f}', va='center')

    st.pyplot(fig)

    # Show result
    if prediction == 1:
        st.error("⚠️ Your profile closely matches diabetic cases in the dataset. Please consult a professional.")
    else:
        st.success("✅ Your profile does not strongly match diabetes patterns in the dataset.")
