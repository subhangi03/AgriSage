import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import joblib

# Load the trained model
model = joblib.load(r'C:\Users\subha\Desktop\AgriSage\Notebooks\crop_model.pkl')
#loading the preprocessor and label encoder
preprocessor = joblib.load(r'C:\Users\subha\Desktop\AgriSage\Notebooks\preprocessor.pkl')
lbl_encoder = joblib.load(r'C:\Users\subha\Desktop\AgriSage\Notebooks\label_encoder.pkl')

# Streamlit app title
st.title('AgriSage')

# Input fields for features
st.header('Input the following features:')
nitrogen = st.number_input('Nitrogen')
phosphorus = st.number_input('Phosphorus')
potassium = st.number_input('Potassium')
temperature = st.number_input('Temperature (°C)')
humidity = st.number_input('Humidity (%)')
ph_value = st.number_input('pH Value')
rainfall = st.number_input('Rainfall (mm)')

# Predict button
if st.button('Predict'):
    # Create a DataFrame with input data
    input_data = pd.DataFrame({
        'Nitrogen': [nitrogen],
        'Phosphorus': [phosphorus],
        'Potassium': [potassium],
        'Temperature': [temperature],
        'Humidity': [humidity],
        'pH_Value': [ph_value],
        'Rainfall': [rainfall]
    })

    # Apply the preprocessor to input_data
    input_data_transformed = preprocessor.transform(input_data)

    # Get prediction probabilities
    prediction_proba = model.predict_proba(input_data_transformed)

    # Get the top 5 probable crops
    top_5_indices = np.argsort(prediction_proba[0])[::-1][:5]
    top_5_probs = prediction_proba[0][top_5_indices]
    top_5_crops = lbl_encoder.inverse_transform(top_5_indices)

    # Display the top 5 probable crops with their probabilities
    st.header('Top 5 Predicted Crops:')
    for crop, prob in zip(top_5_crops, top_5_probs):
        st.write(f'{crop}: {prob:.2%}')

    # Plot the top 5 probable crops with their probabilities
    fig, ax = plt.subplots()
    ax.barh(top_5_crops, top_5_probs, color='skyblue')
    ax.set_xlabel('Probability')
    ax.set_title('Top 5 Predicted Crops')
    ax.invert_yaxis()  # Invert y-axis to display the highest probability at the top

    # Display the plot in Streamlit
    st.pyplot(fig)
