import streamlit as st
import pickle
import numpy as np


def load_rf_model():
    with open('random_forest_degree_pred.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_rf_model()

rf = data["model"]
label_encode_country = data["label_encode_country"]
label_encode_income = data["label_encode_income"]
label_encode_age = data["label_encode_age"]
label_encode_work_hours = data["label_encode_work_hours"]
label_encode_marital_status = data["label_encode_marital_status"]
label_encode_workclass = data["label_encode_workclass"]
label_encode_occupation = data["label_encode_occupation"]

def predict_education():
    st.title("Education Prediction")

    st.write("""### Please enter your informations to predict your Education level""")

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

    income_types = (
        "Less than 50k",
        "Greater than 50k"
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

    income = st.selectbox("Income Level", income_types)
    native_country = st.selectbox("Country", native_countries)
    marraige_status = st.selectbox("Marital Status", marital_status)
    work = st.selectbox("Field of Work", work_types)
    occupation = st.selectbox("Occupation", occupation_types)

    age_group = st.select_slider("Select your age Group", options=['Below 20', '20 - 30', '31 - 40', '41 - 50', '51- 60', 'Above 60'])
    work_hours = st.select_slider("Select your average hours per week", options=['Below 20', '20 - 40', '40 - 60', '60 - 80', 'Above 80'])

    ok = st.button("Predict Education")
    if ok:
        X = np.array([[native_country, income, age_group, work_hours, marraige_status, work, occupation ]])

        X[:, 0] = label_encode_country.transform(X[:,0])
        X[:, 1] = label_encode_income.transform(X[:,1])
        X[:, 2] = label_encode_age.transform(X[:,2])
        X[:, 3] = label_encode_work_hours.transform(X[:,3])
        X[:, 4] = label_encode_marital_status.transform(X[:,4])
        X[:, 5] = label_encode_workclass.transform(X[:,5])
        X[:, 6] = label_encode_occupation.transform(X[:,6])
        X = X.astype(float)

        degree = rf.predict(X)
        if degree[0] == 0:
            st.subheader(f"Minimum qualification needed for this combination is less than a Bachelor's degree")
            st.markdown("""#### The above prediction is based on the adult census dataset. For higher income try improving your standards on any of the features.""")
        elif degree[0] == 1:
            st.subheader(f"Minimum qualification needed for this combination is a Bachelor's degree")
            st.markdown("""#### The above prediction is based on the adult census dataset. For higher income try improving your standards on any of the features.""")
        elif degree[0] == 2:
            st.subheader(f"Minimum qualification needed for this combination is a Master''s degree")
            st.markdown("""#### The above prediction is based on the adult census dataset. For higher income try improving your standards on any of the features.""")
        else:
            st.subheader(f"Minimum qualification needed for this combination is a Ph.d or a Doctorate degree")
            st.markdown("""#### The above prediction is based on the adult census dataset. For higher income try improving your standards on any of the features.""")
