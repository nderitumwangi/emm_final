# =============================================================
# STREAMLIT APP: EMPLOYEE PERFORMANCE PREDICTOR FOR HR
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn   

# -------------------------------
# 1. Load the trained model
# -------------------------------
model_path = "best_employee_performance_model.joblib"
model = joblib.load(model_path)

st.set_page_config(page_title="Employee Performance Predictor", page_icon="💼", layout="wide")

st.title("💼 Employee Performance Prediction App")
st.markdown("""
This app predicts an **employee's performance rating** based on various job and personal attributes.  
Use it to **support hiring and internal performance evaluations**.
""")


