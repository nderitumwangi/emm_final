# =============================================================
# 💼 Employee Performance Predictor - Streamlit App (Full Schema)
# =============================================================

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sklearn
import sys

# -------------------------------------------------------------
# 1. Page Configuration
# -------------------------------------------------------------
st.set_page_config(page_title="Employee Performance Predictor", page_icon="💼", layout="centered")

st.title("💼 Employee Performance Prediction App")
st.markdown("""
Estimate an **employee's performance rating** based on key HR metrics.  
Fill in the details below and click **Predict Performance**.
""")

# -------------------------------------------------------------
# 2. Environment Info
# -------------------------------------------------------------
st.sidebar.title("⚙️ Environment Info")
st.sidebar.write(f"**Python:** {sys.version.split()[0]}")
st.sidebar.write(f"**scikit-learn:** {sklearn.__version__}")
st.sidebar.write(f"**Streamlit:** {st.__version__}")

# -------------------------------------------------------------
# 3. Load model
# -------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = Path("best_employee_performance_model.joblib")
    if not model_path.exists():
        st.error("❌ Model file not found. Please upload or train a new one.")
        st.stop()
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"⚠️ Could not load model: {e}")
        st.stop()

model = load_model()

# -------------------------------------------------------------
# 4. Input Form
# -------------------------------------------------------------
st.subheader("📋 Enter Employee Information")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.slider("Age", 18, 60, 30)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        EducationBackground = st.selectbox("Education Background", [
            "Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"
        ])
        EmpDepartment = st.selectbox("Department", [
            "Sales", "Development", "Research & Development", "Human Resources"
        ])

    with col2:
        EmpJobRole = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Manager", "Laboratory Technician",
            "Developer", "Manufacturing Director", "Healthcare Representative"
        ])
        BusinessTravelFrequency = st.selectbox("Business Travel Frequency", [
            "Non-Travel", "Travel_Rarely", "Travel_Frequently"
        ])
        OverTime = st.selectbox("Overtime", ["Yes", "No"])
        Attrition = st.selectbox("Attrition", ["Yes", "No"])
        EmpEducationLevel = st.slider("Education Level (1–5)", 1, 5, 3)

    with col3:
        EmpJobLevel = st.slider("Job Level (1–5)", 1, 5, 2)
        EmpHourlyRate = st.slider("Hourly Rate", 20, 100, 60)
        EmpRelationshipSatisfaction = st.slider("Relationship Satisfaction (1–5)", 1, 5, 3)
        NumCompaniesWorked = st.slider("Number of Companies Worked", 0, 10, 2)
        DistanceFromHome = st.slider("Distance From Home (km)", 0, 30, 10)

    st.divider()
    st.subheader("🧭 Experience and Performance Factors")
    col4, col5 = st.columns(2)

    with col4:
        EmpJobInvolvement = st.slider("Job Involvement (1–5)", 1, 5, 3)
        EmpEnvironmentSatisfaction = st.slider("Environment Satisfaction (1–5)", 1, 5, 3)
        EmpWorkLifeBalance = st.slider("Work-Life Balance (1–5)", 1, 5, 3)
        EmpJobSatisfaction = st.slider("Job Satisfaction (1–5)", 1, 5, 3)

    with col5:
        EmpLastSalaryHikePercent = st.slider("Last Salary Hike (%)", 0, 25, 12)
        YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 2)
        YearsWithCurrManager = st.slider("Years With Current Manager", 0, 15, 3)
        ExperienceYearsInCurrentRole = st.slider("Years in Current Role", 0, 20, 5)
        ExperienceYearsAtThisCompany = st.slider("Years at Current Company", 0, 20, 6)
        TotalWorkExperienceInYears = st.slider("Total Work Experience (Years)", 0, 40, 10)
        TrainingTimesLastYear = st.slider("Trainings Attended Last Year", 0, 10, 2)

    submitted = st.form_submit_button("🚀 Predict Performance")

# -------------------------------------------------------------
# 5. Make Prediction
# -------------------------------------------------------------
if submitted:
    AgeGroup = (
        "Young" if Age < 30 else
        "Mid-age" if Age <= 45 else
        "Senior"
    )

    # Construct a full schema input DataFrame
    input_data = pd.DataFrame([{
        "Age": Age,
        "Gender": Gender,
        "MaritalStatus": MaritalStatus,
        "EducationBackground": EducationBackground,
        "EmpDepartment": EmpDepartment,
        "EmpJobRole": EmpJobRole,
        "BusinessTravelFrequency": BusinessTravelFrequency,
        "OverTime": OverTime,
        "Attrition": Attrition,
        "EmpEducationLevel": EmpEducationLevel,
        "EmpJobLevel": EmpJobLevel,
        "EmpHourlyRate": EmpHourlyRate,
        "EmpRelationshipSatisfaction": EmpRelationshipSatisfaction,
        "NumCompaniesWorked": NumCompaniesWorked,
        "DistanceFromHome": DistanceFromHome,
        "EmpJobInvolvement": EmpJobInvolvement,
        "EmpEnvironmentSatisfaction": EmpEnvironmentSatisfaction,
        "EmpWorkLifeBalance": EmpWorkLifeBalance,
        "EmpJobSatisfaction": EmpJobSatisfaction,
        "EmpLastSalaryHikePercent": EmpLastSalaryHikePercent,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager,
        "ExperienceYearsInCurrentRole": ExperienceYearsInCurrentRole,
        "ExperienceYearsAtThisCompany": ExperienceYearsAtThisCompany,
        "TotalWorkExperienceInYears": TotalWorkExperienceInYears,
        "TrainingTimesLastYear": TrainingTimesLastYear,
        "AgeGroup": AgeGroup
    }])

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"⭐ Predicted Employee Performance Rating: **{prediction} / 4**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # -------------------------------------------------------------
    # 6. Display Interpretations
    # -------------------------------------------------------------
    if prediction >= 4:
        st.markdown("### 🟢 High Performer — Great potential for leadership or growth roles!")
        st.balloons()
    elif prediction == 3:
        st.markdown("### 🟡 Consistent Performer — Shows reliability and solid contribution.")
    else:
        st.markdown("### 🔴 Needs Support — Focus on engagement, mentoring, and training.")

    st.markdown("---")
    st.markdown("### 💡 Key HR Factors Influencing Performance")
    st.markdown("""
    - 🏆 **TrainingTimesLastYear** – More training correlates with higher performance.  
    - 📈 **YearsSinceLastPromotion** – Long gaps can reduce motivation.  
    - 👔 **ExperienceYearsAtThisCompany** – Mid-tenure (3–7 years) tends to perform best.  
    - ⚖️ **Work-Life Balance** – Healthy balance drives sustained performance.  
    - 💬 **Job Involvement** – Active engagement is a strong predictor of high ratings.  
    """)
