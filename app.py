import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Page Config
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠")

st.title("🧠 Brain Tumor Detection App")
st.write("Upload an MRI image to detect whether a brain tumor is present.")

# Load Model (cached)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("brain_tumor_model.keras")
    return model

model = load_model()

# Class labels (must match training folders)
class_names = ['No Tumor', 'Tumor']

# Image Preprocessing Function
def preprocess_image(image):
    image = image.resize((128,128))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# File uploader
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader("Prediction Result:")
    st.write(f"**Result:** {class_names[predicted_class]}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")

    if predicted_class == 1:
        st.error("⚠️ Tumor Detected! Please consult a medical professional.")
    else:
        st.success("✅ No Tumor Detected.")
        
