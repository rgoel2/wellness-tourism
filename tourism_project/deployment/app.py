import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier  # Or whatever model type you're using

st.title("Wellness Tourism Package")

# Load the model locally instead of from Hugging Face
# Option 1: If you have the model file locally
try:
    model = joblib.load("wellness_tourism_model.joblib")  # Adjust path as needed
except FileNotFoundError:
    # Option 2: If you don't have the model, create a simple one for demonstration
    st.warning("Model file not found. Creating a simple demonstration model instead.")
    # Create a simple model for demonstration purposes
    model = RandomForestClassifier()
    model.fit(
        [[35, 1, 1, 10, 1, 1, 2, 1, 1, 3, 1, 2, 0, 3, 0, 0, 1, 50000]],  # Sample features
        [0]  # Sample target
    )

customer_data = {
    "Age": st.number_input("Age", min_value=18, max_value=100, value=35),
    "TypeofContact": st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"]),
    "CityTier": st.selectbox("City Tier", ["1", "2", "3"]),
    "DurationOfPitch": st.number_input("Duration of Pitch", min_value=1, max_value=60, value=10),
    "Occupation": st.selectbox("Occupation", ["Salaried", "Free Lancer", "Business", "Student", "Others"]),
    "Gender": st.selectbox("Gender", ["Male", "Female"]),
    "NumberOfPersonVisiting": st.number_input("Number of People Visiting", min_value=1, max_value=10, value=2),
    "NumberOfFollowups": st.number_input("Number of Followups", min_value=0, max_value=10, value=1),
    "ProductPitched": st.selectbox("Product Pitched", ["Basic", "Deluxe", "Super Deluxe"]),
    "PreferredPropertyStar": st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3),
    "MaritalStatus": st.selectbox("Marital Status", ["Single", "Married", "Divorced"]),
    "NumberOfTrips": st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=2),
    "Passport": st.selectbox("Passport", [0, 1]),
    "PitchSatisfactionScore": st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3),
    "OwnCar": st.selectbox("Own Car", [0, 1]),
    "NumberOfChildrenVisiting": st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0),
    "Designation": st.selectbox("Designation", ["Manager", "Executive", "Salaried", "Free Lancer", "Others"]),
    "MonthlyIncome": st.number_input("Monthly Income", min_value=1000.0, max_value=200000.0, value=50000.0),
}

# Convert categorical variables to numeric if needed
input_df = pd.DataFrame([customer_data])
# You might need to add preprocessing here to match the model's expected input format

if st.button("Predict Purchase Likelihood"):
    try:
        proba = model.predict_proba(input_df)[0, 1]
        prediction = "likely to purchase" if proba >= 0.5 else "unlikely to purchase"
        st.write(f"Predicted outcome: **{prediction}**")
        st.write(f"Purchase probability: {proba:.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.info("You may need to preprocess the input data to match the model's expected format.")
