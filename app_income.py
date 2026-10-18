import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
# Note: Ensure 'income_model.pkl' is saved in the same folder from your Income Classification notebook
try:
    model, scaler = joblib.load('income_model.pkl')
except Exception as e:
    st.error(f"Model file not found! Please run your income classification training notebook first. Error: {e}")

st.title("💼 Income Classification Prediction App")
st.write("Enter the individual's demographic and employment details below to predict if their income exceeds $50K/year.")

# Input fields categorized for a clean user experience
st.subheader("Demographic Information")
age = st.number_input("Age", value=35, min_value=17, max_value=90, step=1)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
race = st.selectbox("Race", options=[0, 1, 2, 3, 4], format_func=lambda x: ["Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"][x])
native_country = st.selectbox("Native Country (Encoded Index)", options=range(0, 40), index=39) # Default to US approx index

st.subheader("Employment & Occupation")
workclass = st.selectbox("Workclass", options=range(0, 8), format_func=lambda x: ["Federal-gov", "Local-gov", "Never-worked", "Private", "Self-emp-inc", "Self-emp-not-inc", "State-gov", "Without-pay"][x])
occupation = st.selectbox("Occupation", options=range(0, 14), format_func=lambda x: f"Occupation Category {x}")
hours_per_week = st.number_input("Hours Worked Per Week", value=40, min_value=1, max_value=100, step=1)
fnlwgt = st.number_input("Final Weight (fnlwgt)", value=200000.0)

st.subheader("Education & Marital Status")
education_num = st.slider("Education Number (Years of Education)", min_value=1, max_value=16, value=10)
marital_status = st.selectbox("Marital Status", options=range(0, 7), format_func=lambda x: f"Marital Status Category {x}")
relationship = st.selectbox("Relationship", options=range(0, 6), format_func=lambda x: f"Relationship Category {x}")

st.subheader("Financial Metrics")
capital_gain = st.number_input("Capital Gain", value=0.0, step=100.0)
capital_loss = st.number_input("Capital Loss", value=0.0, step=100.0)

# Prediction button
if st.button("Predict Income Category"):
    try:
        # Arrange features in the exact order used during training
        # Order: age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country
        # Note: If your training dataset encoded 'education' as a separate categorical column alongside 'education-num', 
        # ensure this array matches the number of columns your X train had.
        features = np.array([[age, workclass, fnlwgt, education_num, education_num, 
                              marital_status, occupation, relationship, race, sex, 
                              capital_gain, capital_loss, hours_per_week, native_country]])
        
        # Scale features and make prediction
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        
        # Map prediction back to text label if encoded as 0 and 1
        result_label = ">50K" if prediction[0] == 1 else "<=50K"
        
        st.success(f"🎯 Predicted Income Category: **{result_label}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")