### Instead of running the prediction in a .ipynb file, we are using Streamlit to present it in the web

##importing the required libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

##Loading the model
model=tf.keras.models.load_model("model.h5")

##Loading the encoder and scalar
with open('label_encoder.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('onehot_encoder.pkl','rb') as file:
    OneHot_encoder_geo=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


    ##Streamlit

st.title("Customer Churn Prediction")

##Input Data
geography = st.selectbox('Geography', OneHot_encoder_geo.categories_[0])###Dropdown,(cat[0]) will be the default
gender = st.selectbox('Gender', label_encoder_gender.classes_)##Male or female
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


##Preparing the input_data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = OneHot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=OneHot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
print(input_data)
##Here since we don't have geography in the input_data df, we can use reset_index()
#reset_index() replaces previous df index with the new one, if drop=True
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
print(input_data)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')


