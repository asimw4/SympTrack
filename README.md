# 🫀 SympTrack: AI-Powered Heart Health Insight Tool

🚀 [Live App on Streamlit](https://symptrack.streamlit.app)

SympTrack is an interactive web tool that allows users to input key health metrics and receive an AI-powered heart disease risk insight. Built with Streamlit and a trained logistic regression model on real-world heart disease data.

## Project Purpose

This project demonstrates the application of interpretable machine learning to a healthcare use case. The goal is to provide an AI-powered tool that not only predicts risk similarity to heart disease cases, but also explains the key factors driving each prediction — a critical need in clinical and digital health settings.

## Features

- Predicts **heart disease risk similarity** based on UCI Heart Disease dataset (Cleveland subset)
- Interactive **health metric inputs**
- Visualizes **top factors influencing each prediction**
- Full **feature contribution bar chart** for transparency
- **Model metrics:**  
  - Accuracy: 88%  
  - F1 Score: 86%  
  (F1 Score balances precision and recall — crucial for healthcare applications)

## Potential Applications

- Prototype for **clinical trial pre-screening tools**
- Early-stage **digital health app** concept
- Demonstration of **explainable AI in healthcare**

## How It Works

- A **logistic regression model** was trained on 13 features from the UCI dataset.
- User inputs are processed and fed to the model in real time.
- Outputs include:
  - A **risk similarity score**
  - **Top 3 contributing factors**
  - Full feature contribution visualization

## Future Improvements

- Test additional models (e.g. XGBoost, Random Forest) and compare performance
- Add model calibration (important in clinical applications)
- Enhance explainability with SHAP-based visualization
- Deploy with HTTPS / custom domain

## Disclaimer

This tool is for **educational purposes only** and does not provide medical advice.

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- Matplotlib

---

**Created by Asim Waheed**

