import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
try:
    model, scaler = joblib.load('life_expectancy_model.pkl')
except Exception as e:
    st.error(f"Model file not found! Please run your life expectancy training notebook first. Error: {e}")

st.title("🌍 Life Expectancy Prediction App")
st.write("Enter the country's demographic, health, and economic parameters below to estimate Life Expectancy.")

# Input fields aligned with dataset features
st.subheader("Demographic & Country Details")
year = st.number_input("Year", value=2023, step=1)
status = st.selectbox("Status", options=[0, 1], format_func=lambda x: "Developed (0)" if x == 0 else "Developing (1)")
country = st.number_input("Country (Encoded Index)", value=0, step=1)

st.subheader("Health & Mortality Factors")
adult_mortality = st.number_input("Adult Mortality Rates per 1,000 population", value=150.0)
infant_deaths = st.number_input("Infant Deaths per 1,000 population", value=5.0)
alcohol = st.number_input("Alcohol Consumption (litres per capita)", value=4.0)
percentage_expenditure = st.number_input("Percentage Expenditure on health (% of GDP)", value=50.0)
hepatitis_b = st.number_input("Hepatitis B immunization coverage (%)", value=80.0)
measles = st.number_input("Measles cases per 1,000 population", value=10.0)
bmi = st.number_input("Average BMI", value=25.0)
under_five_deaths = st.number_input("Under-five deaths per 1,000 population", value=5.0)
polio = st.number_input("Polio immunization coverage (%)", value=80.0)
total_expenditure = st.number_input("General Government Expenditure on Health (%)", value=6.0)
diphtheria = st.number_input("Diphtheria immunization coverage (%)", value=80.0)
hiv_aids = st.number_input("HIV/AIDS deaths per 1,000 live births", value=0.1)

st.subheader("Economic & Social Factors")
gdp = st.number_input("Gross Domestic Product (GDP in USD)", value=1000.0)
population = st.number_input("Total Population", value=1000000.0)
thinness_1to19 = st.number_input("Thinness 10-19 years (%)", value=4.0)
thinness_5to9 = st.number_input("Thinness 5-9 years (%)", value=4.0)
income_composition = st.number_input("Income Composition of Resources (0 to 1)", value=0.6, min_value=0.0, max_value=1.0)
schooling = st.number_input("Schooling (Number of years)", value=12.0)

# Prediction button
if st.button("Predict Life Expectancy"):
    try:
        # Match exact feature column order used in training dataframe (X)
        features = np.array([[country, year, status, adult_mortality, infant_deaths, alcohol,
                              percentage_expenditure, hepatitis_b, measles, bmi, under_five_deaths,
                              polio, total_expenditure, diphtheria, hiv_aids, gdp, population,
                              thinness_1to19, thinness_5to9, income_composition, schooling]])
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        
        st.success(f"🎯 Estimated Life Expectancy: **{prediction[0]:.2f} years**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")