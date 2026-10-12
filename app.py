import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Lung Capacity Predictor",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 Lung Capacity Predictor")
st.write("Enter the details to predict lung capacity.")

# Load the trained model from the same folder as app.py
MODEL_PATH = Path(__file__).parent / "lung_cap_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


# Load model safely
try:
    model = load_model()

except FileNotFoundError:
    st.error(
        "Model file 'lung_cap_model.pkl' was not found. "
        "Please keep it in the same folder as app.py."
    )
    st.stop()

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# Input section
st.subheader("Enter Input Details")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=25
)

height = st.number_input(
    "Height (cm)",
    min_value=50.0,
    max_value=250.0,
    value=165.0
)

smoke = st.selectbox("Smoke", ["No", "Yes"])
gender = st.selectbox("Gender", ["Male", "Female"])
caesarean = st.selectbox("Caesarean", ["No", "Yes"])


# Prediction
if st.button("Predict Lung Capacity"):

    try:
        # Encode categorical values
        smoke_value = 1 if smoke == "Yes" else 0
        gender_value = 1 if gender == "Male" else 0
        caesarean_value = 1 if caesarean == "Yes" else 0

        # Prepare input features
        input_data = pd.DataFrame(
            [[
                age,
                height,
                smoke_value,
                gender_value,
                caesarean_value
            ]],
            columns=[
                "Age",
                "Height",
                "Smoke",
                "Gender",
                "Caesarean"
            ]
        )

        # Generate prediction
        prediction = np.asarray(
            model.predict(input_data)
        ).reshape(-1)[0]

        # Display result
        st.success("Prediction completed!")

        st.metric(
            "Predicted Lung Capacity",
            f"{float(prediction):.2f}"
        )

        st.caption(
            "This is a model estimate, not a medical diagnosis."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.write(
            "Please check the feature names, category encoding, "
            "and preprocessing used during model training."
        )