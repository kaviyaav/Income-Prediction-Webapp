import streamlit as st
import pickle
import numpy as np


def load_rf_model():
    with open('random_forest_income_pred.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_rf_model()

rf = data["model"]
label_encode_country = data["label_encode_country"]
label_encode_edu = data["label_encode_edu"]
label_encode_age = data["label_encode_age"]
label_encode_work_hours = data["label_encode_work_hours"]
label_encode_marital_status = data["label_encode_marital_status"]
label_encode_workclass = data["label_encode_workclass"]
label_encode_occupation = data["label_encode_occupation"]

def predict_income():
    st.title("Income Prediction")

    st.write("""### Please enter your information to predict your approximate income slab""")

    native_countries = (
        "United-States",
        "Mexico",
        "Canada",
        "India",
        "Philippines",
        "Germany",
        "Puerto-Rico",
        "El-Salvador",
        "Others"
    )

    education_types = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Ph.d or Doctorate degree",
    )

    marital_status = (
        "Never-married",
        "Widowed",
        "Divorced",
        "Separated",
        "Married-civ-spouse",
        "Married-spouse-absent",
        "Married-AF-spouse"
    )

    work_types = (
        "Private",
        "State Government",
        "Federal Government",
        "Local Government",
        "Self employed"
    )

    occupation_types = (
        "Executive Manager",
        "Professional specialist ",
        "Craft Occupation",
        "Admin clerical",
        "Sales",
        "Other Service"
    )

    education_level = st.selectbox("Education Level", education_types)
    native_country = st.selectbox("Country", native_countries)
    marraige_status = st.selectbox("Marital Status", marital_status)
    work = st.selectbox("Field of Work", work_types)
    occupation = st.selectbox("Occupation", occupation_types)

    age_group = st.select_slider("Select your age Group", options=['Below 20', '20 - 30', '31 - 40', '41 - 50', '51- 60', 'Above 60'])
    work_hours = st.select_slider("Select your average hours per week", options=['Below 20', '20 - 40', '40 - 60', '60 - 80', 'Above 80'])

    ok = st.button("Predict Income")
    if ok:
        X = np.array([[native_country, education_level, age_group, work_hours, marraige_status, work, occupation ]])

        X[:, 0] = label_encode_country.transform(X[:,0])
        X[:, 1] = label_encode_edu.transform(X[:,1])
        X[:, 2] = label_encode_age.transform(X[:,2])
        X[:, 3] = label_encode_work_hours.transform(X[:,3])
        X[:, 4] = label_encode_marital_status.transform(X[:,4])
        X[:, 5] = label_encode_workclass.transform(X[:,5])
        X[:, 6] = label_encode_occupation.transform(X[:,6])
        X = X.astype(float)

        income = rf.predict(X)
        if income[0] == 0:
            st.subheader(f"You will fall under average salary slab of 50k")
            st.markdown("""#### The above prediction is based on the adult census dataset. For higher income try improving your standards on any of the features.""")
        else:
            st.subheader(f"Your will fall above average salary slab of 50k")
            st.markdown("""#### Congratulations you fall under higher income slab!!!""")
            st.markdown("""#### The above prediction is based on the adult census dataset. For higher income try improving your standards on any of the features.""")
