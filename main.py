import streamlit as st

# Import tab renderers (we'll build these next)
from heart_tab import render as render_heart
from diabetes_tab import render as render_diabetes
from hypertension_tab import render as render_hypertension

# Page setup
st.set_page_config(page_title="SympTrack", layout="wide")
st.title("🧬 SympTrack: AI-Powered Metabolic Health Dashboard")
st.markdown("Predict your risk for Heart Disease, Diabetes, or Hypertension.")

# Sidebar radio to select tab
selected_tab = st.sidebar.radio("Select Condition:", ["Heart Disease", "Diabetes", "Hypertension"])

# Sidebar info panel
if selected_tab == "Heart Disease":
    st.sidebar.title("ℹ️ About SympTrack")
    st.sidebar.markdown("""
    **Model:** Logistic Regression  
    **Accuracy:** 88%  
    **F1 Score:** 86%  
    **Dataset:** UCI Heart Disease (Cleveland subset)  

    This tool is for **educational purposes only** and does not provide medical advice.
    """)
    render_heart()

elif selected_tab == "Diabetes":
    st.sidebar.title("ℹ️ About SympTrack")
    st.sidebar.markdown("""
    **Model:** Random Forest  
    **Accuracy:** 77%  
    **F1 Score:** 63%  
    **Dataset:** Pima Indians Diabetes Dataset  

    This tool is for **educational purposes only** and does not provide medical advice.
    """)
    render_diabetes()

elif selected_tab == "Hypertension":
    st.sidebar.title("ℹ️ About SympTrack")
    st.sidebar.markdown("""
    **Model:** Random Forest 
    **Accuracy:** 72%  
    **F1 Score:** 84%  
    **Dataset:** Repurposed Stroke Dataset for Hypertension Risk  

    This tool is for **educational purposes only** and does not provide medical advice.
    """)
    render_hypertension()


