import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ===============================
# PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Employee Performance Prediction App")
st.markdown("""
This application predicts **employee performance** based on key HR factors.  
Use it to assess potential new hires or identify employees needing support or development.
""")

# ===============================
# LOAD TRAINED MODEL
# ===============================
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "best_employee_performance_model.joblib")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# ===============================
# USER INPUT FORM
# ===============================
st.sidebar.header("📋 Input Employee Details")

# Numeric Inputs
age = st.sidebar.slider("Age", 18, 60, 30)
distance = st.sidebar.slider("Distance From Home (km)", 1, 30, 10)
education_level = st.sidebar.slider("Education Level", 1, 5, 3)
environment_satisfaction = st.sidebar.slider("Environment Satisfaction", 1, 4, 3)
job_involvement = st.sidebar.slider("Job Involvement", 1, 4, 3)
job_satisfaction = st.sidebar.slider("Job Satisfaction", 1, 4, 3)
num_companies = st.sidebar.slider("Number of Companies Worked", 0, 10, 3)
last_hike = st.sidebar.slider("Last Salary Hike (%)", 0, 25, 10)
relationship = st.sidebar.slider("Relationship Satisfaction", 1, 4, 3)
total_experience = st.sidebar.slider("Total Work Experience (Years)", 0, 40, 10)
training_times = st.sidebar.slider("Training Times Last Year", 0, 10, 2)
work_life_balance = st.sidebar.slider("Work-Life Balance", 1, 4, 3)
experience_company = st.sidebar.slider("Years at Company", 0, 40, 5)
experience_role = st.sidebar.slider("Years in Current Role", 0, 20, 5)
years_promo = st.sidebar.slider("Years Since Last Promotion", 0, 20, 3)
years_manager = st.sidebar.slider("Years With Current Manager", 0, 20, 4)

# Categorical Inputs
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
department = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
job_role = st.sidebar.selectbox("Job Role", [
    "Manager", "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative", "Human Resources"
])
education_bg = st.sidebar.selectbox("Education Background", [
    "Marketing", "Life Sciences", "Medical", "Technical Degree", "Human Resources", "Other"
])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
travel = st.sidebar.selectbox("Business Travel Frequency", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])

# Combine all inputs into a DataFrame
input_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "EducationBackground": education_bg,
    "MaritalStatus": marital_status,
    "EmpDepartment": department,
    "EmpJobRole": job_role,
    "BusinessTravelFrequency": travel,
    "DistanceFromHome": distance,
    "EmpEducationLevel": education_level,
    "EmpEnvironmentSatisfaction": environment_satisfaction,
    "EmpJobInvolvement": job_involvement,
    "EmpJobSatisfaction": job_satisfaction,
    "NumCompaniesWorked": num_companies,
    "OverTime": overtime,
    "EmpLastSalaryHikePercent": last_hike,
    "EmpRelationshipSatisfaction": relationship,
    "TotalWorkExperienceInYears": total_experience,
    "TrainingTimesLastYear": training_times,
    "EmpWorkLifeBalance": work_life_balance,
    "ExperienceYearsAtThisCompany": experience_company,
    "ExperienceYearsInCurrentRole": experience_role,
    "YearsSinceLastPromotion": years_promo,
    "YearsWithCurrManager": years_manager
}])

# ===============================
# DISPLAY INPUTS
# ===============================
st.subheader("👤 Input Summary")
st.dataframe(input_data)

# ===============================
# PREDICTION
# ===============================
if st.button("🔍 Predict Performance"):
    prediction = model.predict(input_data)[0]
    st.markdown("## 🧾 Prediction Result")
    st.success(f"Predicted Employee Performance Rating: **{prediction}** (out of 4)")

    # Conditional recommendations
    if prediction <= 2:
        st.warning("""
        **Performance Insight: Low Performance Expected**
        - Focus on targeted training and mentoring.
        - Monitor job satisfaction and workload balance.
        - Enhance career development opportunities.
        """)
    elif prediction == 3:
        st.info("""
        **Performance Insight: Average Performer**
        - Maintain consistent engagement through recognition.
        - Offer skill enhancement programs and career growth paths.
        """)
    else:
        st.success("""
        **Performance Insight: High Performer**
        - Excellent potential for leadership and advanced roles.
        - Retention strategies recommended (bonuses, new challenges).
        """)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("© 2025 INX Future Inc | Employee Performance Prediction | Streamlit Deployment")
