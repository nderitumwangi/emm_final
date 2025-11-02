import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn

# Load the model
@st.cache_resource
def load_model():
    return joblib.load("best_employee_performance_model_sklearn172.joblib")

def main():
    st.title("Employee Performance Predictor")
    st.write("Enter employee information to predict performance rating")

    # Create input fields
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=70, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        department = st.selectbox("Department", ["Sales", "Development", "Data Science", 
                                               "Research & Development", "Human Resources", "Finance"])
        job_role = st.selectbox("Job Role", ["Sales Executive", "Developer", "Data Scientist",
                                           "Research Scientist", "HR", "Financial Analyst"])
        distance_from_home = st.number_input("Distance From Home", min_value=0, max_value=100, value=10)
        
    with col2:
        education = st.selectbox("Education Level", ["Below College", "College", "Bachelor", "Master", "Doctor"])
        environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
        work_life_balance = st.slider("Work Life Balance", 1, 4, 3)
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
        years_in_role = st.number_input("Years in Current Role", min_value=0, max_value=20, value=3)

    # Create a dataframe from inputs
    data = {
        'Age': age,
        'Gender': gender,
        'EmpDepartment': department,
        'EmpJobRole': job_role,
        'DistanceFromHome': distance_from_home,
        'EmpEducationLevel': education,
        'EmpEnvironmentSatisfaction': environment_satisfaction,
        'EmpJobSatisfaction': job_satisfaction,
        'EmpWorkLifeBalance': work_life_balance,
        'ExperienceYearsAtThisCompany': years_at_company,
        'ExperienceYearsInCurrentRole': years_in_role
    }
    input_df = pd.DataFrame([data])

    if st.button("Predict Performance"):
        # Load model and make prediction
        model = load_model()
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)

        # Display prediction
        st.subheader("Prediction Results")
        st.write(f"Predicted Performance Rating: {prediction[0]}")
        
        # Display probabilities
        st.write("Prediction Probabilities:")
        prob_df = pd.DataFrame(probability[0], 
                             columns=['Probability'],
                             index=['Rating 2', 'Rating 3', 'Rating 4'])
        st.dataframe(prob_df)

        # Add interpretation
        if prediction[0] == 4:
            st.success("Outstanding Performance! This employee shows excellent potential.")
        elif prediction[0] == 3:
            st.info("Good Performance. This employee meets expectations.")
        else:
            st.warning("Performance needs improvement. Consider additional support and training.")

if __name__ == "__main__":
    main()