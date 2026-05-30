import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Page configuration
st.set_page_config(page_title="Handwritten Digit Recognition", layout="wide")
st.title("Handwritten Digit Recognition")

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_model.h5")

model = load_model()

# Layout: two columns
col1, col2 = st.columns([1, 1])

# File uploader
uploaded_file = col1.file_uploader("Upload a digit image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open image and convert to grayscale
    image = Image.open(uploaded_file).convert("L")
    
    # Display uploaded image
    col1.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to numpy array
    img_gray = np.array(image)

    # Invert colors if background is white
    if np.mean(img_gray) > 127:
        img_gray = 255 - img_gray

    # Resize while keeping aspect ratio to 20x20
    h, w = img_gray.shape
    if h > w:
        new_h = 20
        new_w = int(w * (20 / h))
    else:
        new_w = 20
        new_h = int(h * (20 / w))
    img_resized = cv2.resize(img_gray, (new_w, new_h))

    # Add padding to center the digit in 28x28
    pad_top = (28 - new_h) // 2
    pad_bottom = 28 - new_h - pad_top
    pad_left = (28 - new_w) // 2
    pad_right = 28 - new_w - pad_left
    img_padded = np.pad(img_resized, ((pad_top, pad_bottom), (pad_left, pad_right)), "constant", constant_values=0)

    # Normalize and reshape
    img_normalized = img_padded / 255.0
    img_input = img_normalized.reshape(1, 28, 28, 1)

    # Predict
    y_pred = model.predict(img_input)[0]

    # Top prediction
    top_idx = np.argmax(y_pred)
    col2.markdown(f"### Predicted Digit: {top_idx} (Probability: {y_pred[top_idx]:.2f})")

    # Hidden list of all predictions
    with col2.expander("See all probabilities"):
        for idx, prob in enumerate(y_pred):
            st.write(f"Digit {idx}: {prob:.2f}")

else:
    st.info("Please upload an image of a handwritten digit to get started.")
