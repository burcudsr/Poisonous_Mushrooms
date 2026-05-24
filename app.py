import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# =====================================================================
# STEP 1: Load Model and Encoders
# =====================================================================

model = load_model("mushroom_final_model.keras")

with open("mushroom_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# =====================================================================
# STEP 2: Streamlit UI Layout
# =====================================================================

st.title("Mushroom Classification AI 🍄")
st.write("Predict whether a mushroom is Edible or Poisonous using a highly accurate Deep Learning model.")

# We will collect all user inputs inside a neat Sidebar
st.sidebar.header("Mushroom Features")

input_data = {}
numerical_cols = ['cap-diameter', 'stem-height', 'stem-width']

# Dynamically generate dropdown selects for categorical features (Filtering out numeric noise)
for col, encoder in encoders.items():
    # 1. Filter out any synthetic numeric noise (0-9) from the category list
    categories = [
        cat for cat in encoder.categories_[0] 
        if not str(cat).isdigit()
    ]
    
    # 2. Sort alphabetically and move 'Missing' to the very end of the list
    if 'Missing' in categories:
        categories.remove('Missing')
        categories.sort()
        categories.append('Missing')
    else:
        categories.sort()
        
    # Fallback safety in case a column ends up completely empty after filtering
    if not categories:
        categories = ['Missing']
        
    # Render the clean, sorted dropdown options to the user
    selected_val = st.sidebar.selectbox(label=col, options=categories, index=0)
    input_data[col] = selected_val

# Add numerical input boxes with default baseline values
input_data['cap-diameter'] = st.sidebar.number_input("cap-diameter", value=9.0)
input_data['stem-height'] = st.sidebar.number_input("stem-height", value=6.0)
input_data['stem-width'] = st.sidebar.number_input("stem-width", value=15.0)

# =====================================================================
# STEP 3: Prediction Logic
# =====================================================================

if st.button("Predict Safety", type="primary"):
    encoded_dict = {}
    numeric_values = []
    
    # 1. Process categorical columns into the model's 'input_xxx' format
    for col, value in input_data.items():
        if col in encoders:
            encoder = encoders[col]
            try:
                encoded_val = encoder.transform([[value]])[0][0]
            except:
                encoded_val = 0
            
            # Match the Keras expected input naming pattern: 'cap-shape' -> 'input_cap_shape'
            keras_input_name = f"input_{col.replace('-', '_')}"
            encoded_dict[keras_input_name] = np.array([encoded_val])
        else:
            # Collect physical measurements for the single numeric block
            numeric_values.append(float(value))
            
    # 2. Combine numerical features into a single matrix under 'input_numeric'
    encoded_dict['input_numeric'] = np.array([numeric_values])
            
    # Get direct probability output from the deep learning model
    prediction = model.predict(encoded_dict)
    probability = float(prediction[0][0])
    
    # Render the structured results UI
    st.write("---")
    if probability > 0.5:
        st.error(f"### Result: Poisonous ⚠️")
        st.write(f"Confidence score that this mushroom is toxic: **{probability:.2%}**")
    else:
        st.success(f"### Result: Edible 🍽️")
        st.write(f"Confidence score that this mushroom is safe: **{(1 - probability):.2%}**")