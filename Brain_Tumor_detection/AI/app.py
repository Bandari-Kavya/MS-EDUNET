import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("brain_tumor_model.h5")  # or .keras
image_size = 128

st.title("🧠 Brain Tumor Detector")

uploaded = st.file_uploader("Upload MRI image", type=["jpg","png","jpeg"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, use_column_width=True)

    img_arr = img.resize((image_size, image_size))
    img_arr = np.expand_dims(img_arr, axis=0) / 255.0
    pred = model.predict(img_arr)[0][0]

    st.subheader("Result:")
    if pred > 0.5:
        st.error("Tumor Detected")
    else:
        st.success("No Tumor Detected")
