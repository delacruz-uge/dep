import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Constants
TARGET_SIZE = 180

# Load model
@st.cache_resource
def load_my_model():
    return load_model("model.h5")

model = load_my_model()

# Preprocess uploaded image
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((TARGET_SIZE, TARGET_SIZE))
    image = np.array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# UI
st.title("Bird vs. Drone Classifier")
st.write("Upload an image to classify it as a bird or a drone.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_names = ['Bird', 'Drone']
    predicted_class = class_names[np.argmax(prediction)]

    st.write("### Prediction:", predicted_class)
    st.write("Confidence:", f"{np.max(prediction)*100:.2f}%")
