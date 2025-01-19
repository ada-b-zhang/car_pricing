import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64

# Wallpaper
with open("wallpaper.jpg", "rb") as image_file:  
    base64_image = base64.b64encode(image_file.read()).decode()

st.markdown(
    f"""
    <style>
    body {{
        background-image: url("data:image/jpg;base64,{base64_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stApp {{
        background: rgba(255, 255, 255, 0.5);
        padding: 10px;
        border-radius: 15px;
    }}
    
    /* Success box styling */
    .stAlert {{
        background-color: #28a745 !important; 
        color: white !important; 
        border-radius: 10px !important;
        padding: 10px !important;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1) !important;
    }}

    /* Button styling */
    div.stButton > button {{
        background-color: #09b530 !important; 
        color: white !important; /* White text */
        border-radius: 8px !important;
        padding: 0.5em 1em !important;
        font-size: 16px !important;
        border: none !important;
    }}
    div.stButton > button:hover {{
        background-color: #04751d !important; 
        color: white !important;
    }}
    div.stButton > button:active {{
        background-color: #0e591f !important; 
        color: white !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model
try:
    model = joblib.load('model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the scaler
try:
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    st.stop()

# Load the label encoders
try:
    label_encoders = joblib.load('label_encoders.pkl')
except Exception as e:
    st.error(f"Error loading label encoders: {e}")
    st.stop()

# Define numeric and non-numeric features
numeric_features = ['model_year', 'mileage', 'horsepower', 'engine_size'] 
non_numeric_features = ['make', 'car_model', 'ext_col', 'int_col', 'accident', 'transmission_type']

# Categorical choices for dropdowns
with open("make_choices.txt", "r") as file:
    data = file.read()  
    make_list = eval(data)

with open("car_model_choices.txt", "r") as file:
    data = file.read() 
    car_model_list = eval(data)

with open("ext_col_choices.txt", "r") as file:
    data = file.read() 
    ext_col_list = eval(data)

with open("int_col_choices.txt", "r") as file:
    data = file.read()
    int_col_list = eval(data)

make_choices = sorted(make_list)
car_model_choices = sorted(car_model_list)
ext_col_choices = sorted(ext_col_list) 
int_col_choices = sorted(int_col_list) 
accident_choices = ["No", "Yes"]
transmission_choices = ["Automatic", "Manual"]

# Map dropdown selections to encoded integers
dropdown_mappings = {
    "make": make_choices,
    "model": car_model_choices,
    "ext_col": ext_col_choices,
    "int_col": int_col_choices,
    "accident": accident_choices,
    "transmission_type": transmission_choices
}

st.markdown(
    """
    <h1 style='text-align: center;'>
    What is the Value of Your Car? 🚙
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <h4 style='text-align: center; font-size:18px;'>
    Estimate the value of your car with artificial intelligence. 
    </h4>
    """,
    unsafe_allow_html=True
)

# User Inputs 
make = st.selectbox("Make", options=make_choices)
car_model = st.selectbox("Model", options=car_model_choices)
model_year = st.number_input("Year", min_value=1900, max_value=2025, step=1, value=2020)
mileage = st.number_input("Mileage", min_value=0.0, max_value=1e6, step=100.0, value=50000.0)
transmission_type = st.selectbox("Transmission Type", options=transmission_choices)
ext_col = st.selectbox("Exterior Color", options=ext_col_choices)
int_col = st.selectbox("Interior Color", options=int_col_choices)
accident = st.selectbox("Has the car been in an accident?", options=accident_choices)
horsepower = st.number_input("Horsepower", min_value=0.0, max_value=2000.0, step=10.0, value=150.0)
engine_size = st.number_input("Engine Size (liters)", min_value=0.0, max_value=10.0, step=0.1, value=2.0)

# Prediction button
if st.button("Predict Price"):
    # Map dropdown selections to encoded integers
    mapped_inputs = {
        "make": make_choices.index(make),
        "model": car_model_choices.index(car_model),
        "ext_col": ext_col_choices.index(ext_col),
        "int_col": int_col_choices.index(int_col),
        "accident": accident_choices.index(accident),
        "transmission_type": transmission_choices.index(transmission_type)
    }

    # Prepare the input DataFrame
    input_features = pd.DataFrame([[
        model_year, mileage, 
        horsepower, engine_size,
        mapped_inputs["make"], mapped_inputs["model"], 
        mapped_inputs["ext_col"], mapped_inputs["int_col"],
        mapped_inputs["accident"], mapped_inputs["transmission_type"]
    ]], columns=numeric_features + non_numeric_features)

    # Scale numeric features
    try:
        scaled_numeric = pd.DataFrame(
            scaler.transform(input_features[numeric_features]),
            columns=numeric_features
        )
        input_features_scaled = pd.concat([scaled_numeric, input_features[non_numeric_features]], axis=1)
    except Exception as e:
        st.error(f"Error scaling numeric features: {e}")
        st.stop()

    # Encode non-numeric features
    try:
        for feature in non_numeric_features:
            encoder = label_encoders.get(feature)
            if encoder:
                input_features_scaled[feature] = input_features_scaled[feature].map(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )
    except Exception as e:
        st.error(f"Error encoding non-numeric features: {e}")
        st.stop()

    # Make prediction
    try:
        log_prediction = model.predict(input_features_scaled)[0]  # Remember that model predicts in log scale!!!
        predicted_price = np.expm1(log_prediction)  # Reverse log transformation
        # st.success(f"Estimated Price: ${predicted_price:,.2f}")
        st.markdown(
            f"""
            <div style='
                background-color: #28a745; 
                color: white !important;
                border-radius: 5px; 
                padding: 5px; 
                text-align: center; 
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); 
                font-family: Arial, sans-serif;'>
                <h1 style='font-size: 40px; font-weight: bold; margin: 0;'>${predicted_price:,.2f}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")


