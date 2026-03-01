import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠")
st.title("🧠 Brain Tumor Detection App")
st.write("Upload an MRI image to detect whether a brain tumor is present.")

# -------------------------------
# Load Model (Cached)
# -------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("brain_tumor_model.keras")
    return model

model = load_model()

# -------------------------------
# Class Names (Change if needed)
# -------------------------------
class_names = ['No Tumor', 'Tumor']  # Make sure order matches training

# -------------------------------
# Image Preprocessing Function
# -------------------------------
def preprocess_image(image, target_size=(128, 128)):
    image = image.resize(target_size)
    image = np.array(image)

    # Normalize if model was trained with rescale=1./255
    image = image / 255.0  

    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # Display Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    processed_image = preprocess_image(image)

    # Prediction
    prediction = model.predict(processed_image)

    st.subheader("Prediction Details")
    st.write("Raw Prediction Output:", prediction)

    # -------------------------------
    # Automatic Handling of Sigmoid / Softmax
    # -------------------------------
    if prediction.shape[1] == 1:
        # Sigmoid output
        prob = prediction[0][0]

        if prob > 0.5:
            predicted_class = 1
            confidence = prob
        else:
            predicted_class = 0
            confidence = 1 - prob

    else:
        # Softmax output
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

    # -------------------------------
    # Display Result
    # -------------------------------
    st.subheader("Prediction Result")
    st.write(f"**Result:** {class_names[predicted_class]}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    if predicted_class == 1:
        st.error("⚠️ Tumor Detected! Please consult a medical professional.")
    else:
        st.success("✅ No Tumor Detected.")
