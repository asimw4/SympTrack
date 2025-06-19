# 🩺 SympTrack: AI-Powered Health Risk Insight Tool

🚀 [Live App on Streamlit](https://symptrack.streamlit.app/)

SympTrack is a multi-condition AI-powered web app that provides health risk insights for Heart Disease, Diabetes, and Hypertension. Users can input real-world health metrics and receive predictions along with transparent explanations of contributing factors — all in real time.

---

## 🎯 Project Purpose

This project demonstrates the use of interpretable machine learning for health risk prediction across three major conditions. The goal is to provide transparency and insight alongside predictions — a key requirement in responsible clinical and digital health applications.

---

## ✨ Features

- Predicts risk similarity for:
  - **Heart Disease**
  - **Diabetes**
  - **Hypertension**
- Interactive metric-based UI with dropdowns, sliders, and toggles
- Top 3 influencing features per prediction
- Full feature contribution chart for transparency
- Custom-trained Logistic Regression models
- Tailored inputs and features per condition

---

## 📊 Model Metrics

### Heart Disease:
- Accuracy: 88%
- F1 Score: 86%

### Diabetes:
- Accuracy: 76%
- F1 Score: 82%

### Hypertension:
- Accuracy: 72%
- F1 Score: 84%

*(F1 Score balances precision and recall — especially important in health predictions.)*

---

## 🔍 How It Works

- Each condition uses its own logistic regression model trained on real-world datasets.
- User inputs are encoded to match model expectations.
- For every prediction:
  - A **Risk Similarity Score** is shown
  - The **Top 3 Risk Factors** are listed
  - A **Feature Impact Chart** explains the model's reasoning

---

## 🔬 Potential Applications

- Digital health app prototypes
- Clinical trial pre-screening
- Explainable AI demonstrations in healthcare
- Risk education tools for public health

---

## 📈 Future Improvements

- Swap logistic regression for ensemble models (e.g. XGBoost)
- Integrate SHAP or LIME for model-agnostic explainability
- Add support for more conditions (e.g. stroke, kidney disease)
- Enable user account storage + condition history
- Custom domain deployment with HTTPS

---

## ⚠️ Disclaimer

**This tool is for educational purposes only. It does not provide medical advice.** Always consult a medical professional for real health guidance.

---

## 🧰 Tech Stack

- **Languages**: Python  
- **Frameworks**: Streamlit  
- **ML Libraries**: Scikit-learn  
- **Visualization**: Matplotlib  
- **Modeling Techniques**: Logistic Regression  
- **Version Control**: Git + GitHub  
- **Deployment**: Streamlit Cloud  

---

## 👨‍💻 Created by
**Asim Waheed**  
[GitHub](https://github.com/asimw4) | [LinkedIn](https://linkedin.com/in/your-link)

