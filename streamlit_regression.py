import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Load trained model
model = tf.keras.models.load_model('regression_model.h5')

# Load encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit App
st.title("Employee Salary Prediction")

st.header("Enter Customer Details")

# Inputs
credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.slider(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

tenure = st.slider(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

num_of_products = st.slider(
    "Number Of Products",
    min_value=1,
    max_value=4,
    value=2
)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

exited = st.selectbox(
    "Exited",
    [0, 1]
)

geography = st.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)

# Create DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

# One Hot Encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Merge Geography Columns
input_data = pd.concat(
    [input_data.reset_index(drop=True),
     geo_encoded_df.reset_index(drop=True)],
    axis=1
)

# Ensure Correct Column Order
input_data = input_data.reindex(columns=[
    'CreditScore',
    'Gender',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'IsActiveMember',
    'Exited',
    'Geography_France',
    'Geography_Germany',
    'Geography_Spain'
])

# Scale Features
input_data_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_data_scaled)

predicted_salary = prediction[0][0]

# Display Result
st.subheader("Predicted Salary")

st.success(
    f"Estimated Salary: ${predicted_salary:,.2f}"
)