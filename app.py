import streamlit as st
from PIL import Image
import numpy as np
import time

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Rare Disease X-ray Project - Day 2",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. IMAGE PREPROCESSING UTILITIES
# ==========================================
def preprocess_image(image: Image.Image):
    """
    Basic preprocessing: Resize to 224x224 and normalize placeholder.
    """
    # Resize to standard model input size
    img_resized = image.resize((224, 224))
    
    # Convert to numpy array
    img_array = np.array(img_resized).astype('float32')
    
    # Placeholder Normalization (e.g., scale to [0, 1])
    img_normalized = img_array / 255.0
    
    return img_normalized

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🩺 Diagnostic Menu")
app_mode = st.sidebar.selectbox(
    "Choose Application Mode",
    ["User Mode", "Admin Mode"]
)

st.sidebar.markdown("---")
st.sidebar.write("**Project Status:** Day 2 Development")

# ==========================================
# 4. USER MODE: DIAGNOSTIC DASHBOARD
# ==========================================
if app_mode == "User Mode":
    st.title("🩻 Chest X-Ray Analysis")
    st.write("Upload a patient's chest X-ray for automated classification and attention mapping.")
    
    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload Chest X-Ray (PNG, JPG, JPEG)", 
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file:
        # Load and Display Image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Patient X-Ray")
            st.image(image, use_container_width=True, caption="Original Upload")
            
        with col2:
            st.subheader("Diagnostic Controls")
            if st.button("🚀 Run Prediction"):
                with st.spinner("Preprocessing and Analyzing..."):
                    # Simulate processing time
                    processed_data = preprocess_image(image)
                    time.sleep(1.5) 
                    
                    # Dummy Prediction Results
                    prediction = "Pneumonia Detected"
                    confidence = 0.942
                    
                    # Display Prediction Output
                    st.success(f"**Result:** {prediction}")
                    st.metric(label="Confidence Score", value=f"{confidence:.2%}")
                    
                    st.info("Note: This is a placeholder prediction for Day 2 development.")
            
            st.markdown("---")
            st.subheader("🔍 Grad-CAM Visualization")
            
            # Grad-CAM Placeholder Logic
            heatmap_available = False # Toggle this when model is integrated
            
            if heatmap_available:
                st.image("https://via.placeholder.com/400x400.png?text=Grad-CAM+Heatmap", caption="Attention Map")
            else:
                st.info("Grad-CAM Heatmap will be generated upon successful prediction.")
                st.image("https://via.placeholder.com/400x400.png?text=Heatmap+Placeholder", use_container_width=True)

    else:
        st.info("Waiting for image upload...")

# ==========================================
# 5. ADMIN MODE: PROJECT METADATA
# ==========================================
else:
    st.title("🛡️ Admin Control Center")
    st.write("System configuration and training pipeline overview.")
    
    # Metadata Section
    st.header("📊 Project Metadata")
    meta_col1, meta_col2 = st.columns(2)
    
    with meta_col1:
        st.markdown(f"""
        - **Dataset Name:** ChestX-ray14 (Modified)
        - **Number of Classes:** 14 Diseases
        - **Model Architecture:** EfficientNet-B4
        """)
        
    with meta_col2:
        st.markdown(f"""
        - **Diffusion Model:** Conditional DDPM
        - **Evaluation Metric:** F1-Score (Macro)
        - **Target Accuracy:** > 92%
        """)
    
    st.markdown("---")
    
    # Pipeline Section
    st.header("⚙️ Training Pipeline")
    st.markdown("""
    1. **Data Acquisition:** Sourcing from NIH Clinical Center.
    2. **Preprocessing:** CLAHE enhancement and 224x224 resizing.
    3. **Augmentation:** Using Conditional DDPM to balance rare disease classes.
    4. **Feature Extraction:** Transfer learning with EfficientNet backbone.
    5. **Fine-tuning:** End-to-end training on balanced dataset.
    6. **Evaluation:** Cross-validation and Grad-CAM verification.
    """)
    
    st.success("Pipeline configuration verified for Day 2.")

# ==========================================
# 6. FOOTER
# ==========================================
st.markdown("---")
st.caption("Rare Disease X-ray Project | Day 2 Update | Confidential Medical Data")
