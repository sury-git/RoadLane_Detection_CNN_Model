import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Load trained lane detection model
model = tf.keras.models.load_model("lane_detection_unet.keras")

# Function to perform lane detection
def detect_lanes(image):
    img_resized = cv2.resize(image, (256, 256)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension
    pred_mask = model.predict(img_resized)[0]  # Predict lane mask
    pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Binarize
    return pred_mask * 255

# Streamlit UI
st.title("Lane Detection Web App 🚗🛣️")
uploaded_file = st.file_uploader("Upload a Road Image", type=["jpg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    lane_mask = detect_lanes(image)
    
    # Overlay lane mask on the original image
    overlayed = cv2.addWeighted(image, 0.7, cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)

    st.image(overlayed, caption="Detected Lanes", use_column_width=True)
