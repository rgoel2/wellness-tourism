import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.title("Wellness Tourism Package")

try:
    model_path = hf_hub_download(repo_id="novicetopper/wellness-tourism-model", filename="wellness_tourism_model.joblib" )
    model = joblib.load(model_path)
except FileNotFoundError:
    st.warning("Model file not found.")
    
Age = st.number_input("Age", min_value=18, max_value=100, value=35)
TypeofContact= st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
CityTier= st.selectbox("City Tier", ["1", "2", "3"])
DurationOfPitch= st.number_input("Duration of Pitch", min_value=1, max_value=60, value=10)
Occupation= st.selectbox("Occupation", ["Salaried", "Free Lancer", "Business", "Student", "Others"])
Gender= st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting= st.number_input("Number of People Visiting", min_value=1, max_value=10, value=2)
NumberOfFollowups= st.number_input("Number of Followups", min_value=0, max_value=10, value=1)
ProductPitched= st.selectbox("Product Pitched", ["Basic", "Deluxe", "Super Deluxe"])
PreferredPropertyStar= st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
MaritalStatus= st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips= st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=2)
Passport= st.selectbox("Passport", [0, 1])
PitchSatisfactionScore= st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
OwnCar= st.selectbox("Own Car", [0, 1])
NumberOfChildrenVisiting= st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
Designation= st.selectbox("Designation", ["Manager", "Executive", "Salaried", "Free Lancer", "Others"])
MonthlyIncome= st.number_input("Monthly Income", min_value=1000.0, max_value=200000.0, value=50000.0)

# Convert categorical variables to numeric if needed
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier" : CityTier,
    "DurationOfPitch" : DurationOfPitch,
    "Occupation" : Occupation,
    "Gender" : Gender,
    "NumberOfPersonVisiting" : NumberOfPersonVisiting,
    "NumberOfFollowups" : NumberOfFollowups,
    "ProductPitched" : ProductPitched,
    "PreferredPropertyStar" : PreferredPropertyStar,
    "MaritalStatus" : MaritalStatus,
    "NumberOfTrips" : NumberOfTrips,
    "Passport" : Passport,
    "PitchSatisfactionScore" : PitchSatisfactionScore,
    "OwnCar" : OwnCar,
    "NumberOfChildrenVisiting" : NumberOfChildrenVisiting,
    "Designation" : Designation,
    "MonthlyIncome": MonthlyIncome
}])


if st.button("Predict Purchase Likelihood"):
        proba = model.predict_proba(input_data)[0, 1]
        prediction = "likely to purchase" if proba >= 0.5 else "unlikely to purchase"
        st.write(f"Predicted outcome: **{prediction}**")
        st.write(f"Purchase probability: {proba:.2f}")