import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model/car_price_model.pkl")

st.title("🚗 Car Price Prediction App")

# User inputs
year = st.number_input("Year", 2000, 2023, 2018)
mileage = st.number_input("Mileage (km)", 0, 300000, 40000)
engine = st.number_input("Engine Size (L)", 1.0, 5.0, 1.6)
horsepower = st.number_input("Horsepower", 50, 500, 130)
fuel = st.selectbox("Fuel Type", ["Petrol","Diesel"])

# Predict
if st.button("Predict Price"):
    fuel_diesel = 1 if fuel=="Diesel" else 0
    df_input = pd.DataFrame({
        'Year':[year],
        'Mileage':[mileage],
        'EngineSize':[engine],
        'Horsepower':[horsepower],
        'FuelType_Diesel':[fuel_diesel]
    })
    
    price = model.predict(df_input)[0]
    st.success(f"Estimated Car Price: ${price:,.0f}")
