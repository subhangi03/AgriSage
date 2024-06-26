import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import random

# Load the trained model
model = joblib.load(r'C:\Users\subha\Desktop\AgriSage\StreamlitApp\crop_model.pkl')
# Load the preprocessor and label encoder
preprocessor = joblib.load(r'C:\Users\subha\Desktop\AgriSage\StreamlitApp\preprocessor.pkl')
lbl_encoder = joblib.load(r'C:\Users\subha\Desktop\AgriSage\StreamlitApp\label_encoder.pkl')

# Load the test data
x_test = pd.read_csv(r'C:\Users\subha\Desktop\AgriSage\StreamlitApp\x_test.csv')
y_test = pd.read_csv(r'C:\Users\subha\Desktop\AgriSage\StreamlitApp\y_test.csv')

# Streamlit app title
st.title('AgriSage')

# Layout with two columns
col1, col2 = st.columns(2)

with col1:
    # Input fields for features
    st.header('Input the following features:')
    
    st.write('Nitrogen:')
    nitrogen = st.number_input('', value=0, min_value=0, max_value=100, key='nitrogen_number_input')
    nitrogen_slider = st.slider('', min_value=0, max_value=100, value=nitrogen, key='nitrogen_slider')
    
    st.write('Phosphorus:')
    phosphorus = st.number_input('', value=0, min_value=0, max_value=100, key='phosphorus_number_input')
    phosphorus_slider = st.slider('', min_value=0, max_value=100, value=phosphorus, key='phosphorus_slider')
    
    st.write('Potassium:')
    potassium = st.number_input('', value=0, min_value=0, max_value=100, key='potassium_number_input')
    potassium_slider = st.slider('', min_value=0, max_value=100, value=potassium, key='potassium_slider')
    
    st.write('Temperature (°C):')
    temperature = st.number_input('', value=0.0, min_value=0.0, max_value=50.0, key='temperature_number_input')
    temperature_slider = st.slider('', min_value=0.0, max_value=50.0, value=temperature, key='temperature_slider')
    
    st.write('Humidity (%):')
    humidity = st.number_input('', value=0.0, min_value=0.0, max_value=100.0, key='humidity_number_input')
    humidity_slider = st.slider('', min_value=0.0, max_value=100.0, value=humidity, key='humidity_slider')
    
    st.write('pH Value:')
    ph_value = st.number_input('', value=0.0, min_value=0.0, max_value=14.0, key='ph_value_number_input')
    ph_value_slider = st.slider('', min_value=0.0, max_value=14.0, value=ph_value, key='ph_value_slider')
    
    st.write('Rainfall (mm):')
    rainfall = st.number_input('', value=0.0, min_value=0.0, max_value=300.0, key='rainfall_number_input')
    rainfall_slider = st.slider('', min_value=0.0, max_value=300.0, value=rainfall, key='rainfall_slider')

    # Use slider values if adjusted, otherwise use number input values
    nitrogen = nitrogen_slider
    phosphorus = phosphorus_slider
    potassium = potassium_slider
    temperature = temperature_slider
    humidity = humidity_slider
    ph_value = ph_value_slider
    rainfall = rainfall_slider

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

        with col2:
            st.header('Results:')
            # Plot the top 5 probable crops with their probabilities
            colors = plt.cm.Paired(np.linspace(0, 1, len(top_5_crops)))
            fig, ax = plt.subplots()
            bars = ax.barh(top_5_crops, top_5_probs, color=colors)
            ax.set_xlabel('Probability')
            ax.set_title('Top 5 Predicted Crops')
            ax.invert_yaxis()  # Invert y-axis to display the highest probability at the top

            # Add percentage labels to the bars
            for bar, prob in zip(bars, top_5_probs):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height() / 2, f'{prob:.2%}', va='center', ha='left')

            # Display the plot in Streamlit
            st.pyplot(fig)

    # Validate button
    if st.button('Validate with Random Test Sample'):
        # Select a random sample from the test set
        random_index = random.randint(0, len(x_test) - 1)
        random_sample = x_test.iloc[random_index:random_index+1]
        actual_value = y_test.iloc[random_index]

        # Apply the preprocessor to the random sample
        random_sample_transformed = preprocessor.transform(random_sample)

        # Get prediction probabilities
        prediction_proba = model.predict_proba(random_sample_transformed)

        # Get the top 5 probable crops
        top_5_indices = np.argsort(prediction_proba[0])[::-1][:5]
        top_5_probs = prediction_proba[0][top_5_indices]
        top_5_crops = lbl_encoder.inverse_transform(top_5_indices)

        # Get the actual crop
        actual_crop = lbl_encoder.inverse_transform([actual_value])[0]

        with col2:
            st.header('Results:')
            # Plot the top 5 probable crops with their probabilities
            colors = plt.cm.Paired(np.linspace(0, 1, len(top_5_crops)))
            fig, ax = plt.subplots()
            bars = ax.barh(top_5_crops, top_5_probs, color=colors)
            ax.set_xlabel('Probability')
            ax.set_title('Top 5 Predicted Crops')
            ax.invert_yaxis()  # Invert y-axis to display the highest probability at the top

            # Add percentage labels to the bars
            for bar, prob in zip(bars, top_5_probs):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height() / 2, f'{prob:.2%}', va='center', ha='left')

            # Display the plot in Streamlit
            st.pyplot(fig)

            # Display the actual and predicted crop
            st.write(f"Actual Crop: {actual_crop}")
            st.write(f"Top Predicted Crop: {top_5_crops[0]}")