import numpy as np
import pandas as pd
import joblib

import sklearn
import streamlit as st

st.sidebar.write(f"🧩 Scikit-learn version: {sklearn.__version__}")


# --- Load Model and  ---

# --- Streamlit App UI ---
st.title("Employee Performance Prediction")
st.write("Predict employee performance based on various features.")

