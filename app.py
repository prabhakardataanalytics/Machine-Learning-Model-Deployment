

import streamlit as st
import pickle
import pandas as pd

# Load the model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to make predictions
def predict(age, sex, bmi, children, smoker, region):
    # Prepare the input data in the same format as the training data
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })

    input_data = pd.get_dummies(input_data, drop_first=True)

    # Ensure all columns are present
    columns = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']
    for col in columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[columns]

    # Make the prediction
    prediction = model.predict(input_data)[0]
    return prediction

# Streamlit app
st.title('Insurance Charges Prediction')

age = st.number_input('Age', min_value=0, max_value=100, value=25)
sex = st.selectbox('Sex', ['male', 'female'])
bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=25.0)
children = st.number_input('Children', min_value=0, max_value=10, value=0)
smoker = st.selectbox('Smoker', ['yes', 'no'])
region = st.selectbox('Region', ['northeast', 'northwest', 'southeast', 'southwest'])

if st.button('Predict'):
    prediction = predict(age, sex, bmi, children, smoker, region)
    st.write(f'Predicted Insurance Charges: ${prediction:.2f}')
