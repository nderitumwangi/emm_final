import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
import os

# Load the model
@st.cache_resource
def load_model():
    # Get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct path to model file
    model_path = os.path.join(script_dir, "best_employee_performance_model.joblib")
    
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}")
        st.info("Please make sure the model file is in the same directory as this script")
        return None

def main():
    st.title("Employee Performance Predictor")
    st.write("Enter employee information to predict performance rating")

    # ... existing code ...

    if st.button("Predict Performance"):
        # Load model and make prediction
        model = load_model()
        
        if model is not None:
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