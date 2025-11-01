# ============================================================
# 🎯 Employee Performance Predictor - Streamlit Web App
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
import sys
from pathlib import Path

# ------------------------------------------------------------
# 1. Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🎯 Employee Performance Predictor")
st.markdown("""
This interactive tool helps HR professionals estimate an **employee's performance rating**
based on behavioral and workplace factors.  
Just fill in the details below and click **Predict Performance**.
""")

# ------------------------------------------------------------
# 2. Display Environment Info
# ------------------------------------------------------------
st.sidebar.title("⚙️ Environment Info")
st.sidebar.write(f"**Python:** {sys.version.split()[0]}")
st.sidebar.write(f"**Streamlit:** {st.__version__}")
st.sidebar.write(f"**scikit-learn:** {sklearn.__version__}")

# ------------------------------------------------------------
# 3. Load the Trained Model (cached)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = Path("best_employee_performance_model.joblib")
    if not model_path.exists():
        st.error("❌ Model file not found. Please retrain and save as 'best_employee_performance_model.joblib'.")
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {e}")
        return None

model = load_model()

# ------------------------------------------------------------
# 4. Input Form
# ------------------------------------------------------------
st.subheader("📋 Enter Employee Information")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        EmpLastSalaryHikePercent = st.slider("Last Salary Hike (%)", 0, 25, 12)
        EmpEnvironmentSatisfaction = st.slider("Environment Satisfaction (1–5)", 1, 5, 3)
        EmpJobSatisfaction = st.slider("Job Satisfaction (1–5)", 1, 5, 3)

    with col2:
        YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 2)
        EmpWorkLifeBalance = st.slider("Work-Life Balance (1–5)", 1, 5, 3)
        EmpJobInvolvement = st.slider("Job Involvement (1–5)", 1, 5, 3)

    submitted = st.form_submit_button("🚀 Predict Performance")

# ------------------------------------------------------------
# 5. Make Prediction
# ------------------------------------------------------------
if submitted:
    if model is None:
        st.error("❌ Model not loaded. Please retrain or check the file.")
    else:
        # Prepare input data
        input_data = pd.DataFrame({
            'EmpLastSalaryHikePercent': [EmpLastSalaryHikePercent],
            'EmpEnvironmentSatisfaction': [EmpEnvironmentSatisfaction],
            'EmpJobSatisfaction': [EmpJobSatisfaction],
            'YearsSinceLastPromotion': [YearsSinceLastPromotion],
            'EmpWorkLifeBalance': [EmpWorkLifeBalance],
            'EmpJobInvolvement': [EmpJobInvolvement]
        })

        # Make prediction safely
        try:
            prediction = model.predict(input_data)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # ------------------------------------------------------------
        # 6. Display Prediction
        # ------------------------------------------------------------
        st.success(f"⭐ Predicted Performance Rating: **{prediction:.2f} / 4.0**")

        # Performance interpretation
        if prediction >= 3.5:
            st.markdown("### 🟢 Excellent Performance Potential!")
            st.balloons()
            st.caption("Consider for leadership, mentorship, or high-growth roles.")
        elif prediction >= 3.0:
            st.markdown("### 🟡 Good Performer")
            st.caption("Solid performer. Offer targeted training and growth opportunities.")
        else:
            st.markdown("### 🔴 Needs Development Support")
            st.caption("Provide coaching, skill training, and close performance tracking.")

        # ------------------------------------------------------------
        # 7. HR Insights
        # ------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 💡 Key HR Insights (Based on Model Feature Importance)")
        st.markdown("""
        - 🏆 **Training & Engagement**: Frequent training correlates with higher ratings.  
        - 📈 **Promotion Timeliness**: Long promotion gaps often signal lower motivation.  
        - ⚖️ **Work-Life Balance**: Stable balance tends to drive consistent performance.  
        - 💬 **Job Involvement**: Active engagement is a strong predictor of success.
        """)
        st.caption("Model trained on INX Future Inc. historical employee data.")
        