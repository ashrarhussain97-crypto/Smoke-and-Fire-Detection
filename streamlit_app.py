import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# Page configuration
st.set_page_config(
    page_title="Fire & Smoke Detection System",
    page_icon="🔥",
    layout="wide"
)

# Title and description
st.title("🔥 Fire & Smoke Detection System")
st.markdown("Upload an image to detect fire or smoke using AI")

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('models/fire_detection_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Make sure 'fire_detection_model.h5' is in the 'models/' folder")
        return None

# Load the model
model = load_model()

# Sidebar
with st.sidebar:
    st.header("About")
    st.info(
        "This application uses deep learning to detect fire and smoke in images. "
        "Upload an image and the AI will analyze it for potential fire hazards."
    )
    st.header("How to Use")
    st.markdown("""
    1. Click 'Browse files' to upload an image
    2. Supported formats: JPG, JPEG, PNG
    3. Wait for the AI to analyze
    4. View the prediction results
    """)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload an image to detect fire or smoke"
)

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Analysis Results")
        
        if model is not None:
            # Preprocess the image
            with st.spinner('Analyzing image...'):
                # Convert PIL image to array
                img_array = np.array(image)
                
                # Resize to model input size (adjust size based on your model)
                img_resized = cv2.resize(img_array, (224, 224))
                
                # Normalize pixel values
                img_normalized = img_resized / 255.0
                
                # Add batch dimension
                img_batch = np.expand_dims(img_normalized, axis=0)
                
                # Make prediction
                prediction = model.predict(img_batch, verbose=0)
                
                # Get confidence score
                confidence = float(prediction[0][0])
                
                # Determine result (adjust threshold as needed)
                threshold = 0.5
                
                if confidence > threshold:
                    st.error("🔥 **FIRE/SMOKE DETECTED!**")
                    st.metric("Confidence", f"{confidence * 100:.2f}%")
                    st.warning("⚠️ Potential fire hazard detected in the image!")
                else:
                    st.success("✅ **NO FIRE/SMOKE DETECTED**")
                    st.metric("Confidence", f"{(1 - confidence) * 100:.2f}%")
                    st.info("The image appears to be safe.")
                
                # Show progress bar for visualization
                st.progress(confidence if confidence > threshold else (1 - confidence))
        else:
            st.error("⚠️ Model not loaded. Cannot make predictions.")

else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload an image to begin detection")
    
    # Example section
    st.markdown("---")
    st.subheader("📊 What This App Does")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔥 Fire Detection")
        st.write("Identifies flames and fire in images")
    
    with col2:
        st.markdown("### 💨 Smoke Detection")
        st.write("Detects smoke patterns and haze")
    
    with col3:
        st.markdown("### 🎯 High Accuracy")
        st.write("AI-powered deep learning model")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Fire & Smoke Detection System | Powered by TensorFlow & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
