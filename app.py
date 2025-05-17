import streamlit as st
import io
import cv2
from PIL import Image
import numpy as np
import os
import tensorflow as tf

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Set up the page layout
st.set_page_config(page_title="Screw Anomaly Detector", page_icon=":nut_and_bolt:")

# --- App Title and Description ---
st.title("Screw Anomaly Detector")

st.caption(
    "Boost Your Quality Control with AI-Powered Screw Inspection"
)

st.write(
    "Upload an image or use your camera to check if a screw is Normal or has an Anomaly."
)

# --- Sidebar Content ---
with st.sidebar:
    st.subheader("About this App")
    st.write(
        "This application uses a trained Artificial Intelligence model to perform visual inspection on screw images."
    )
    
    st.write(
        "The model is trained to classify screw images as either 'Normal' (no defects) or 'Anomaly' (containing defects)."
    )
    
    st.write(
        "Testing images are available in the assets folder (MF, MH, MN, ThS, ThT series)."
    )

# --- Model Loading ---
# Define the path to your SavedModel
MODEL_PATH = './'

# Define the expected image size for the model
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

# Define class names based on your model's training
CLASS_NAMES = ["Anomaly", "Normal"]

@st.cache_resource
def load_tf_model(model_path):
    """Loads the TensorFlow SavedModel."""
    try:
        model = tf.saved_model.load(model_path)
        # Get the inference function from the loaded model
        infer = model.signatures["serving_default"]
        st.sidebar.success("AI Model loaded successfully!")
        return infer
    except Exception as e:
        st.sidebar.error(f"Error loading AI model: {e}")
        return None

# Load the model at the start of the app
infer_function = load_tf_model(MODEL_PATH)

# --- Image Preprocessing Function ---
def preprocess_image(image):
    """Resizes and normalizes the image for model input."""
    # Ensure image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize the image
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    # Convert image to numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255.0)

    # Add a batch dimension
    input_image = np.expand_dims(normalized_image_array, axis=0)
    return input_image

# --- Prediction Function ---
def predict_anomaly(image, infer_fn):
    """Makes a prediction using the loaded TensorFlow model."""
    if infer_fn is None:
        return "Model not loaded", 0.0

    # Preprocess the image
    processed_image = preprocess_image(image)

    try:
        predictions = infer_fn(tf.constant(processed_image))
        output_key = list(predictions.keys())[0]
        prediction_probabilities = predictions[output_key].numpy()[0]

        # Get the predicted class index and confidence
        predicted_class_index = np.argmax(prediction_probabilities)
        confidence = prediction_probabilities[predicted_class_index]
        predicted_class_name = CLASS_NAMES[predicted_class_index]

        return predicted_class_name, confidence

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Prediction Error", 0.0

# --- Image Input Methods ---
st.subheader("Select Image Input Method")
input_method = st.radio(
    "options", ["File Uploader", "Camera Input", "Sample Images"], label_visibility="collapsed"
)

uploaded_file_img = None
camera_file_img = None
sample_img = None

# Check which input method was selected
if input_method == "File Uploader":
    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        uploaded_file_img = Image.open(uploaded_file)
        st.image(uploaded_file_img, caption="Uploaded Image", use_container_width=True)
        st.success("Image uploaded successfully!")
    else:
        st.info("Please upload an image file.")

elif input_method == "Camera Input":
    st.info("Please allow access to your camera.")
    camera_image_file = st.camera_input("Click an Image")
    if camera_image_file is not None:
        camera_file_img = Image.open(camera_image_file)
        st.image(camera_file_img, caption="Camera Input Image", use_container_width=True)
        st.success("Image clicked successfully!")
    else:
        st.info("Please click an image.")

elif input_method == "Sample Images":
    # List sample images from assets folder
    sample_options = ["MF_000.png", "MH_000.png", "MN_000.png", "ThS_000.png", "ThT_000.png"]
    selected_sample = st.selectbox("Select a sample image", sample_options)
    
    if selected_sample:
        try:
            # Construct the path to the sample image
            sample_path = os.path.join("assets", selected_sample)
            if os.path.exists(sample_path):
                sample_img = Image.open(sample_path)
                st.image(sample_img, caption=f"Sample Image: {selected_sample}", use_container_width=True)
                st.success("Sample image loaded successfully!")
            else:
                st.error(f"Sample image not found at path: {sample_path}")
        except Exception as e:
            st.error(f"Error loading sample image: {e}")

# --- Analyse Button and Prediction Display ---
analyse_button = st.button(label="Analyse Screw")

if analyse_button:
    st.subheader("Inspection Result:")
    image_to_predict = None

    if input_method == "File Uploader" and uploaded_file_img is not None:
        image_to_predict = uploaded_file_img
    elif input_method == "Camera Input" and camera_file_img is not None:
        image_to_predict = camera_file_img
    elif input_method == "Sample Images" and sample_img is not None:
        image_to_predict = sample_img
    else:
        st.warning("Please provide an image using one of the input methods.")

    if image_to_predict is not None:
        with st.spinner(text="Analyzing screw image..."):
            predicted_class, confidence = predict_anomaly(image_to_predict, infer_function)

            # Display result with appropriate styling
            st.write(f"**Classification Result:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}")
            
            # Progress bar for confidence visualization
            st.progress(float(confidence))
            
            # User-friendly explanation based on the prediction
            if predicted_class == "Anomaly":
                st.error("**Inspection Status:** Anomaly Detected")
                st.write("This screw appears to have defects or abnormalities.")
            else:
                st.success("**Inspection Status:** Normal")
                st.write("This screw appears to be in good condition, no anomalies detected.")

# --- Additional Instructions ---
st.markdown("---")
st.subheader("How to Use:")
st.write("""
1. Choose an input method (upload an image, use your camera, or select a sample image)
2. Provide a clear image of a screw
3. Click 'Analyse Screw' to get results
4. Review the inspection results to see if an anomaly was detected
""")

# --- Footer ---
st.markdown("---")
st.caption("Powered by TensorFlow and Streamlit")
