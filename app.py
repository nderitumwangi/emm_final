import streamlit as st
import joblib
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore') # Suppress warnings

# --- 1. Load the Model ---
# The pipeline includes all preprocessing steps (imputer, scaler, encoder)
# and the Gradient Boosting Classifier.
try:
    model_path = 'best_employee_performance_model.joblib'
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# --- 2. Define Features and App Layout ---
st.set_page_config(page_title="Employee Performance Predictor")
st.title("👨‍💼 INX Employee Performance Predictor")
st.markdown("Use this app to predict an employee's Performance Rating (3 or 4) based on their features.")

# Define the features to collect for the model
# NOTE: The categorical list should ideally contain all categories for robust OHE
# but this simplified list is based on the notebook's preprocessing pipeline.
# You would need to check the full list of categories used in the training data.

# Features used by the model pipeline (based on the notebook structure)
NUMERIC_COLS = ['Age', 'DistanceFromHome', 'EmpEducationLevel', 'EmpEnvironmentSatisfaction', 
                'EmpHourlyRate', 'EmpJobInvolvement', 'EmpJobLevel', 'EmpJobSatisfaction', 
                'NumCompaniesWorked', 'EmpLastSalaryHikePercent', 'EmpRelationshipSatisfaction', 
                'TotalWorkExperienceInYears', 'TrainingTimesLastYear', 'EmpWorkLifeBalance', 
                'ExperienceYearsAtThisCompany', 'ExperienceYearsInCurrentRole', 
                'YearsSinceLastPromotion', 'YearsWithCurrManager']
                
CATEGORICAL_COLS = ['Gender', 'EducationBackground', 'MaritalStatus', 'EmpDepartment', 
                    'EmpJobRole', 'BusinessTravelFrequency', 'OverTime', 'Attrition']

# Pre-defined options for categorical inputs
CAT_OPTIONS = {
    'Gender': ['Male', 'Female'],
    'EducationBackground': ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'],
    'MaritalStatus': ['Single', 'Married', 'Divorced'],
    'EmpDepartment': ['Sales', 'Development', 'Research & Development', 'Human Resources', 'Finance', 'Data Science'],
    'EmpJobRole': ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 
                   'Manager', 'Sales Representative', 'Research Director', 'Human Resources', 'Finance Manager', 'Data Scientist'],
    'BusinessTravelFrequency': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
    'OverTime': ['Yes', 'No'],
    'Attrition': ['Yes', 'No']
}

# --- 3. Collect User Input ---
st.sidebar.header("Employee Input Features")

def get_user_input():
    # Collect Numeric Inputs in Sidebar
    input_data = {}
    
    st.sidebar.subheader("Numeric Features")
    for col in NUMERIC_COLS:
        # Use a reasonable default value, min, and max based on typical HR data
        if col in ['Age', 'EmpHourlyRate']:
            default_val = 35
            min_val = 18
            max_val = 60
        else:
            default_val = 1
            min_val = 1
            max_val = 30 # For experience/levels
            
        input_data[col] = st.sidebar.number_input(col, min_value=min_val, max_value=max_val, value=default_val, step=1)
    
    st.sidebar.subheader("Categorical Features")
    # Collect Categorical Inputs in Sidebar
    for col in CATEGORICAL_COLS:
        input_data[col] = st.sidebar.selectbox(col, options=CAT_OPTIONS.get(col, []))

    # Convert the dictionary to a DataFrame (must be a single row)
    features = pd.DataFrame([input_data])
    return features

input_df = get_user_input()

st.subheader("Current Employee Data Input:")
st.dataframe(input_df)

# --- 4. Prediction Logic ---
if st.button("Predict Performance Rating"):
    try:
        # The model pipeline expects a DataFrame with the original columns
        prediction = model.predict(input_df)
        
        # The prediction output is either 3 or 4
        result = int(prediction[0])
        
        # Display results
        if result == 4:
            st.success(f"**Predicted Performance Rating: {result} (High)**")
            st.balloons()
        else:
            st.info(f"**Predicted Performance Rating: {result} (Average)**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")