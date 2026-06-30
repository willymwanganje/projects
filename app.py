import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("linear_regression_model.pkl")

st.title("🏠 Household Electricity Consumption Predictor")
st.write("Enter household details below:")

# ------------------------
# BASIC FEATURES
# ------------------------
occupants = st.number_input("Number of Occupants", 1, 20, 3)
appliances = st.number_input("Number of Appliances", 1, 50, 10)
hours = st.number_input("Daily Usage Hours (Avg)", 0.0, 24.0, 6.0)

house_type = st.selectbox(
    "Type of House",
    ["Apartment", "Bungalow", "Single Room", "Self-Contained"]
)

# ------------------------
# APPLIANCES (0/1 FEATURES)
# ------------------------
has_fridge = st.selectbox("Has Fridge", [0, 1])
has_tv = st.selectbox("Has TV", [0, 1])
has_iron = st.selectbox("Has Electric Iron", [0, 1])
has_fan = st.selectbox("Has Fan", [0, 1])
has_ac = st.selectbox("Has Air Conditioner", [0, 1])
has_washing = st.selectbox("Has Washing Machine", [0, 1])
has_heater = st.selectbox("Has Water Heater", [0, 1])
has_microwave = st.selectbox("Has Microwave", [0, 1])
has_kettle = st.selectbox("Has Electric Kettle", [0, 1])
has_phone = st.selectbox("Has Phone Charger", [0, 1])
has_radio = st.selectbox("Has Radio", [0, 1])
has_bulbs = st.selectbox("Has Multiple Lighting Bulbs", [0, 1])

# ------------------------
# PREDICTION
# ------------------------
if st.button("Predict Consumption"):

    input_df = pd.DataFrame([[
        occupants,
        appliances,
        hours,
        has_fridge,
        has_tv,
        has_iron,
        has_fan,
        has_ac,
        has_washing,
        has_heater,
        has_microwave,
        has_kettle,
        has_phone,
        has_radio,
        has_bulbs,
        house_type
    ]], columns=[
        "Number_of_Occupants",
        "Number_of_Appliances",
        "Daily_Usage_Hours_Avg",
        "Has_Fridge",
        "Has_TV",
        "Has_Electric_Iron",
        "Has_Fan",
        "Has_Air_Conditioner",
        "Has_Washing_Machine",
        "Has_Water_Heater",
        "Has_Microwave",
        "Has_Electric_Kettle",
        "Has_Phone_Charger",
        "Has_Radio",
        "Has_Multiple_Lighting_Bulbs",
        "Type_of_House"
    ])

    prediction = model.predict(input_df)[0]

    st.success(f"⚡ Estimated Monthly Consumption: {prediction:.2f} kWh")