import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import easyocr
import torch 

# 🖼️ Title and layout
st.set_page_config(
    page_title="Number Plate Recognition",
    layout="wide"
)
st.title("📷 Number Plate Detection and Recognition")
st.markdown("This app detects number plates from a live camera feed and extracts the text.")

# --- Functions to load models with caching ---
@st.cache_resource
def load_yolo_model(model_path):
    """Loads the YOLOv5 model using torch.hub.load."""
    try:
        # Load the model using the yolov5 library via torch.hub.load
        # The 'source='local'' parameter tells it to look for the repository locally
        # The 'path' parameter points to the .pt file
        model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.info("Please ensure your model file exists and is correctly structured.")
        return None

@st.cache_resource
def load_easyocr_reader():
    """Initializes the EasyOCR reader with caching."""
    try:
        reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='./easyocr_models')
        return reader
    except Exception as e:
        st.error(f"Error loading EasyOCR: {e}")
        return None

# --- Main application logic ---

# The path to your model, relative to the app.py script
model_path = os.path.join("yolov5", "runs", "train", "exp2", "weights", "best.pt")

# Load the models
model = load_yolo_model(model_path)
reader = load_easyocr_reader()

if model is None or reader is None:
    st.stop()

# 📂 Camera input
uploaded_file = st.camera_input("📷 Take a picture of the number plate")

if uploaded_file is not None:
    # Read the image and convert it to a format OpenCV can use
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    # Use the loaded model for inference
    results = model(image_np)
    
    # Render results with bounding boxes
    results.render()
    st.image(results.ims[0], caption="Detected Number Plate", use_column_width=True)
    
    # Process detections for OCR
    boxes_df = results.pandas().xyxy[0]
    if boxes_df.empty:
        st.warning("⚠️ No number plates were detected in the image.")
    else:
        st.subheader("Extracted Number Plate(s):")
        for index, row in boxes_df.iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cropped = image_np[ymin:ymax, xmin:xmax]
            
            st.image(cropped, caption="Cropped Number Plate", width=250)
            
            ocr_result = reader.readtext(cropped)
            
            if ocr_result:
                text = " ".join([res[1] for res in ocr_result])
                st.success(f"🔤 Extracted Text: `{text}`")
            else:
                st.warning("No text found on this number plate.")
