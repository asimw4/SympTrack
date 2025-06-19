import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model + feature columns
hypertension_model = joblib.load("hypertension_model.pkl")
hypertension_columns = joblib.load("hypertension_columns.pkl")

def render():
    st.header("💉 Hypertension Risk")

    # Input fields
    age = st.slider("Age", 10, 100, 40, key="hyper_age")
    bmi = st.slider("BMI", 10.0, 60.0, 25.0, key="hyper_bmi")
    cholesterol = st.slider("Cholesterol", 100, 400, 200, key="hyper_chol")
    sbp = st.slider("Systolic BP", 80, 200, 120, key="hyper_sbp")
    dbp = st.slider("Diastolic BP", 40, 120, 80, key="hyper_dbp")
    glucose = st.slider("Glucose", 50, 200, 100, key="hyper_glucose")
    heart_rate = st.slider("Heart Rate", 40, 150, 75, key="hyper_hr")
    salt = st.selectbox("Salt Intake", ["Low", "Medium", "High"], key="hyper_salt")
    stress = st.selectbox("Stress Level", ["Low", "Medium", "High"], key="hyper_stress")
    activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"], key="hyper_activity")

    # Manual encoding
    salt_encoded = [0, 0] if salt == "Low" else [1, 0] if salt == "Medium" else [0, 1]
    stress_encoded = [0, 0] if stress == "Low" else [1, 0] if stress == "Medium" else [0, 1]
    activity_encoded = [0, 0] if activity == "Low" else [1, 0] if activity == "Moderate" else [0, 1]

    # Assemble DataFrame
    row = [age, bmi, cholesterol, sbp, dbp, glucose, heart_rate] + salt_encoded + stress_encoded + activity_encoded
    columns = [
        "Age", "BMI", "Cholesterol", "Systolic_BP", "Diastolic_BP", "Glucose", "Heart_Rate",
        "Salt_Medium", "Salt_High", "Stress_Medium", "Stress_High",
        "Physical_Moderate", "Physical_High"
    ]
    df_input = pd.DataFrame([row], columns=columns)

    # Fill missing model columns
    for col in hypertension_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[hypertension_columns]

    if st.button("Run Hypertension Prediction"):
        prob = hypertension_model.predict_proba(df_input)[0][1]
        prediction = int(prob >= 0.5)
        percent = round(prob * 100)

        st.markdown(f"🧠 **Risk Similarity Score:** {percent}%")

        # Feature impact
        coefs = hypertension_model.coef_[0]
        impacts = []
        for name, coef, val in zip(hypertension_columns, coefs, df_input.iloc[0]):
            impact = round(abs(coef * val), 4)
            direction = "↑ increases risk" if coef * val > 0 else "↓ decreases risk"
            impacts.append((name, impact, direction, val))

        # Remove features with zero input and zero impact
        filtered_impacts = [(n, i, d) for (n, i, d, v) in impacts if i > 0 and v != 0]

        # Show top features that increase risk
        top_risk_features = [f for f in filtered_impacts if f[2].startswith("↑")]
        top_risk_features = sorted(top_risk_features, key=lambda x: x[1], reverse=True)[:3]

        st.markdown("### 🔍 Top Influencing Factors:")
        if top_risk_features:
            for name, impact, direction in top_risk_features:
                st.write(f"- **{name}** (Impact: {impact}) — *{direction}*")
        else:
            st.write("No major risk-increasing factors detected for this input.")

        # Chart
        st.markdown("### 📊 Full Model Feature Impact")
        names = [n for n, i, d in filtered_impacts]
        values = [i for n, i, d in filtered_impacts]

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(names, values, color="#c0392b")
        ax.invert_yaxis()
        ax.set_xlabel("Impact Score")
        ax.set_title("Feature Contributions to Prediction")

        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{value:.2f}", va='center', ha='left', fontsize=8)

        ax.set_xlim(0, max(values) + 0.05 if values else 0.1)
        st.pyplot(fig)

        if prediction == 1:
            st.error("⚠️ Your profile matches hypertension patterns in the dataset. Consider professional advice.")
        else:
            st.success("✅ Your inputs don’t strongly match known hypertension profiles.")

    with st.expander("🧠 How this works"):
        st.markdown("""
        This logistic regression model estimates hypertension risk  
        using cardiovascular, metabolic, and lifestyle inputs.

        **Common features:**
        - **BP**: Blood pressure  
        - **BMI & Cholesterol**: Metabolic indicators  
        - **Salt, Stress, Activity**: Lifestyle contributions  
        """)

