"""Web form calling the prediction API over HTTP."""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8001").rstrip("/")
st.set_page_config(page_title="California Housing", page_icon="🏠")
st.title("🏠 California Housing Prices")
st.write("Predict the median housing value in a district using the dataset. Room counts, bedroom counts, and population describe the entire district.")
st.caption("Income uses dataset units: approximately tens of thousands of dollars. Predictions reflect the historical data used for training.")

with st.form("housing"):
    left, right = st.columns(2)
    with left:
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-122.23)
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=37.88)
        housing_median_age = st.number_input("Median housing age (years)", min_value=0.0, value=41.0)
        total_rooms = st.number_input("Total rooms in the district", min_value=0, value=880)
        total_bedrooms = st.number_input("Total bedrooms in the district", min_value=0, value=129)
    with right:
        population = st.number_input("District population", min_value=0, value=322)
        households = st.number_input("Number of households", min_value=1, value=126)
        median_income = st.number_input("Median income (dataset units)", min_value=0.0, value=8.3252, format="%.4f")
        ocean_proximity = st.selectbox("Ocean proximity", ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"])
    submitted = st.form_submit_button("Get prediction")

if submitted:
    payload = dict(longitude=longitude, latitude=latitude, housing_median_age=housing_median_age,
                   total_rooms=total_rooms, total_bedrooms=total_bedrooms, population=population,
                   households=households, median_income=median_income, ocean_proximity=ocean_proximity)
    if total_bedrooms > total_rooms:
        st.error("The number of bedrooms cannot exceed the number of rooms.")
    else:
        try:
            with st.spinner("Getting prediction…"):
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
                response.raise_for_status()
                result = response.json()
            st.metric("Median housing value", f"${result['median_house_value']:,.0f}")
        except requests.RequestException:
            st.error("Unable to get a prediction from the API. Check that the service is running and try again.")
