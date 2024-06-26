import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import random

# Load the trained model
model = joblib.load(r'C:\Users\subha\Desktop\AgriSage\Notebooks\crop_model.pkl')
# Load the preprocessor and label encoder
preprocessor = joblib.load(r'C:\Users\subha\Desktop\AgriSage\Notebooks\preprocessor.pkl')
lbl_encoder = joblib.load(r'C:\Users\subha\Desktop\AgriSage\Notebooks\label_encoder.pkl')

# Load the test data
x_test = pd.read_csv(r'C:\Users\subha\Desktop\AgriSage\Notebooks\x_test.csv')
y_test = pd.read_csv(r'C:\Users\subha\Desktop\AgriSage\Notebooks\y_test.csv')

# Streamlit app title
st.title('AgriSage')

# Initialize session state for inputs if not already set
if 'inputs' not in st.session_state:
    st.session_state.inputs = {
        'Nitrogen': 0,
        'Phosphorus': 0,
        'Potassium': 0,
        'Temperature': 0.0,
        'Humidity': 0.0,
        'pH_Value': 0.0,
        'Rainfall': 0.0
    }

# Function to display results
def display_results(top_5_crops, top_5_probs, actual_crop=None):
    with st.sidebar:
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

        # Display the plot
        st.pyplot(fig)

        # Display the actual and predicted crop if available
        if actual_crop:
            st.write(f"Actual Crop: {actual_crop}")
        if top_5_crops:
            st.write(f"Top Predicted Crop: {top_5_crops[0]}")

# Layout with two columns
col1, col2 = st.columns(2)

with col1:
    # Input fields for features
    st.header('Input the following features:')
    
    nitrogen = st.number_input('Nitrogen:', value=st.session_state.inputs['Nitrogen'], min_value=0, max_value=100, key='nitrogen_number_input')
    nitrogen_slider = st.slider('Nitrogen:', min_value=0, max_value=100, value=nitrogen, key='nitrogen_slider')
    
    phosphorus = st.number_input('Phosphorus:', value=st.session_state.inputs['Phosphorus'], min_value=0, max_value=100, key='phosphorus_number_input')
    phosphorus_slider = st.slider('Phosphorus:', min_value=0, max_value=100, value=phosphorus, key='phosphorus_slider')
    
    potassium = st.number_input('Potassium:', value=st.session_state.inputs['Potassium'], min_value=0, max_value=100, key='potassium_number_input')
    potassium_slider = st.slider('Potassium:', min_value=0, max_value=100, value=potassium, key='potassium_slider')
    
    temperature = st.number_input('Temperature (°C):', value=st.session_state.inputs['Temperature'], min_value=0.0, max_value=50.0, key='temperature_number_input')
    temperature_slider = st.slider('Temperature (°C):', min_value=0.0, max_value=50.0, value=temperature, key='temperature_slider')
    
    humidity = st.number_input('Humidity (%):', value=st.session_state.inputs['Humidity'], min_value=0.0, max_value=100.0, key='humidity_number_input')
    humidity_slider = st.slider('Humidity (%):', min_value=0.0, max_value=100.0, value=humidity, key='humidity_slider')
    
    ph_value = st.number_input('pH Value:', value=st.session_state.inputs['pH_Value'], min_value=0.0, max_value=14.0, key='ph_value_number_input')
    ph_value_slider = st.slider('pH Value:', min_value=0.0, max_value=14.0, value=ph_value, key='ph_value_slider')
    
    rainfall = st.number_input('Rainfall (mm):', value=st.session_state.inputs['Rainfall'], min_value=0.0, max_value=300.0, key='rainfall_number_input')
    rainfall_slider = st.slider('Rainfall (mm):', min_value=0.0, max_value=300.0, value=rainfall, key='rainfall_slider')

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

        # Display results
        display_results(top_5_crops, top_5_probs)

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

        # Update input values in session state
        st.session_state.inputs = {
            'Nitrogen': random_sample['Nitrogen'].values[0],
            'Phosphorus': random_sample['Phosphorus'].values[0],
            'Potassium': random_sample['Potassium'].values[0],
            'Temperature': random_sample['Temperature'].values[0],
            'Humidity': random_sample['Humidity'].values[0],
            'pH_Value': random_sample['pH_Value'].values[0],
            'Rainfall': random_sample['Rainfall'].values[0]
        }

        # Display results
        display_results(top_5_crops, top_5_probs, actual_crop)
