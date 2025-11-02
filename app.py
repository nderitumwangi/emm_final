import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load the model
@st.cache_resource
def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "best_employee_performance_model.joblib")
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}")
        return None

def main():
    st.title("Employee Performance Predictor")
    st.write("Enter employee information to predict performance rating")

    # Create input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=70, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        department = st.selectbox("Department", ["Sales", "Development", "Data Science", 
                                               "Research & Development", "Human Resources", "Finance"])
        education = st.selectbox("Education Level", ["Below College", "College", "Bachelor", "Master", "Doctor"])
        education_background = st.selectbox("Education Background", ["Life Sciences", "Medical", "Marketing", 
                                                                   "Technical Degree", "Other"])
        
    with col2:
        job_role = st.selectbox("Job Role", ["Sales Executive", "Developer", "Data Scientist",
                                           "Research Scientist", "HR", "Financial Analyst"])
        job_level = st.number_input("Job Level", min_value=1, max_value=5, value=2)
        distance_from_home = st.number_input("Distance From Home", min_value=0, max_value=100, value=10)
        business_travel = st.selectbox("Business Travel", ["No Travel", "Travel Rarely", "Travel Frequently"])
        overtime = st.selectbox("Over Time", ["Yes", "No"])
        attrition = st.selectbox("Attrition", ["Yes", "No"])

    with col3:
        environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
        relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
        work_life_balance = st.slider("Work Life Balance", 1, 4, 3)
        job_involvement = st.slider("Job Involvement", 1, 4, 3)
        
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
        years_in_role = st.number_input("Years in Current Role", min_value=0, max_value=20, value=3)
        years_with_manager = st.number_input("Years with Current Manager", min_value=0, max_value=20, value=3)
        years_since_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1)
        total_work_years = st.number_input("Total Work Experience (Years)", min_value=0, max_value=40, value=8)
        companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value=10, value=2)
        
        training_times = st.number_input("Training Times Last Year", min_value=0, max_value=10, value=2)
        salary_hike = st.number_input("Last Salary Hike Percent", min_value=0, max_value=25, value=15)
        hourly_rate = st.number_input("Hourly Rate", min_value=20, max_value=100, value=65)

    # Calculate age group
    def get_age_group(age):
        if age < 25: return "18-24"
        elif age < 35: return "25-34"
        elif age < 45: return "35-44"
        else: return "45+"

    # Create a dataframe from inputs
    data = {
        'Age': age,
        'AgeGroup': get_age_group(age),
        'Gender': gender,
        'MaritalStatus': marital_status,
        'EmpDepartment': department,
        'EmpJobRole': job_role,
        'EmpJobLevel': job_level,
        'DistanceFromHome': distance_from_home,
        'EmpEducationLevel': education,
        'EducationBackground': education_background,
        'BusinessTravelFrequency': business_travel,
        'EmpEnvironmentSatisfaction': environment_satisfaction,
        'EmpJobSatisfaction': job_satisfaction,
        'EmpRelationshipSatisfaction': relationship_satisfaction,
        'EmpWorkLifeBalance': work_life_balance,
        'EmpJobInvolvement': job_involvement,
        'ExperienceYearsAtThisCompany': years_at_company,
        'ExperienceYearsInCurrentRole': years_in_role,
        'YearsWithCurrManager': years_with_manager,
        'YearsSinceLastPromotion': years_since_promotion,
        'TotalWorkExperienceInYears': total_work_years,
        'NumCompaniesWorked': companies_worked,
        'TrainingTimesLastYear': training_times,
        'EmpLastSalaryHikePercent': salary_hike,
        'EmpHourlyRate': hourly_rate,
        'OverTime': overtime,
        'Attrition': attrition
    }
    input_df = pd.DataFrame([data])

    if st.button("Predict Performance"):
        model = load_model()
        if model is not None:
            try:
                prediction = model.predict(input_df)
                probability = model.predict_proba(input_df)

                st.subheader("Prediction Results")
                st.write(f"Predicted Performance Rating: {prediction[0]}")
                
                prob_df = pd.DataFrame(probability[0], 
                                     columns=['Probability'],
                                     index=['Rating 2', 'Rating 3', 'Rating 4'])
                st.dataframe(prob_df)

                if prediction[0] == 4:
                    st.success("Outstanding Performance! This employee shows excellent potential.")
                elif prediction[0] == 3:
                    st.info("Good Performance. This employee meets expectations.")
                else:
                    st.warning("Performance needs improvement. Consider additional support and training.")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()