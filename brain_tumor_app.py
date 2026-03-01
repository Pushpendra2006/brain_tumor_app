import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_brain_tumor_model.h5")

model = load_model()

# Load class names dynamically
with open("class_names.json", "r") as f:
    class_indices = json.load(f)

# Convert to proper label order
class_names = list(class_indices.keys())

st.title("🧠 Brain Tumor Classification")
st.write("Upload MRI image to classify tumor type")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    predicted_label = class_names[predicted_index]

    st.success(f"Prediction: {predicted_label}")
    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("Class Probabilities")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]} : {prob*100:.2f}%")