import json
import pickle

import numpy as np
import streamlit as st

# Load model and metadata once
with open("XGB_Model.pkl", "rb") as f:
    model = pickle.load(f)

with open("project_data.json", "r") as f:
    project_data = json.load(f)

REGIONS = ["southwest", "southeast", "northwest", "northeast"]


def predict_insurance(age, sex, bmi, children, smoker, region):
    test_array = np.zeros(model.n_features_in_)
    test_array[0] = age
    test_array[1] = project_data["sex"][sex]
    test_array[2] = bmi
    test_array[3] = children
    test_array[4] = project_data["smoker"][smoker]
    region_col = f"region_{region}"
    region_index = project_data["columns"].index(region_col)
    test_array[region_index] = 1
    return round(model.predict([test_array])[0], 2)


# Page config
st.set_page_config(page_title="Medical Insurance Prediction", page_icon="🏥", layout="centered")
st.title("🏥 Medical Health Insurance Charges Prediction")
st.write("Enter your details below to predict medical insurance charges.")

# Input form
with st.form("insurance_form"):
    age = st.number_input("Age", min_value=1, max_value=150, value=30, step=1)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=1.0, max_value=100.0, value=25.0, step=0.1)
    children = st.number_input("Number of Children", min_value=0, max_value=20, value=0, step=1)
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", REGIONS)

    submitted = st.form_submit_button("Predict Charges")

if submitted:
    try:
        charges = predict_insurance(age, sex, bmi, children, smoker, region)
        st.success(f"Estimated Insurance Charges: **${charges:,.2f}**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
