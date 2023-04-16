import streamlit as st
from predict_income import predict_income
from predict_education import predict_education
from explore_model import explore_income_variations

base = "dark"
backgroundColor="#FFFFFF"

page = st.sidebar.selectbox("Explore Or Predict", ("Predict Income", "Predict Education", "Explore"))

if page == "Predict Income":
    predict_income()
elif page == "Predict Education":
    predict_education()
else:
    explore_income_variations()
