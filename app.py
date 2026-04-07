import streamlit as st
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Chest X-Ray Classifier",
    page_icon="🩻",
    layout="wide"
)

# Custom CSS for Streamlit styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Select Mode", ["User Mode", "Admin Mode"])

if mode == "Admin Mode":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Project Info")
    st.sidebar.info("""
    **Dataset:** ChestX-ray14  
    **Model:** EfficientNet  
    **Diffusion Model:** DDPM
    """)

# Main Content
st.title("Medical Chest X-Ray Classification")
st.write("Upload a chest X-ray image to classify it and view heatmap visualizations.")

# File uploader
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, caption='Uploaded X-Ray', use_container_width=True)
    
    with col2:
        # Prediction Section Placeholder
        st.subheader("Prediction")
        st.warning("Model not integrated yet")
        
        # Grad-CAM Heatmap Section Placeholder
        st.subheader("Grad-CAM Heatmap")
        st.info("Heatmap visualization will appear here")
        # Placeholder for heatmap image
        st.image("https://via.placeholder.com/400x400.png?text=Heatmap+Placeholder", use_container_width=True)

else:
    st.info("Please upload an X-ray image to start the analysis.")

# Footer
st.markdown("---")
st.markdown("© 2026 Medical AI Research Dashboard")

