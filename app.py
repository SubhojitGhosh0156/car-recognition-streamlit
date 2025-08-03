import sys
import os
import torch
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import easyocr

# --- 1. SETUP AND MODEL LOADING ---

# Use the script's directory as the base directory for robust path handling
# This is better than os.getcwd()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(BASE_DIR, 'yolov5')
sys.path.append(YOLO_PATH)

# Use Streamlit's caching to load models only once
@st.cache_resource
def load_yolo_model():
    """Loads the YOLOv5 model from the specified path."""
    model_path = os.path.join(YOLO_PATH, "runs", "train", "exp2", "weights", "best.pt")
    device = select_device('cpu')
    model = DetectMultiBackend(model_path, device=device)
    return model

@st.cache_resource
def load_ocr_reader():
    """Loads the EasyOCR reader."""
    # Using gpu=False is safer for broader compatibility
    return easyocr.Reader(['en'], gpu=False)

# Load the models using the cached functions
yolo_model = load_yolo_model()
ocr_reader = load_ocr_reader()

# --- 2. STREAMLIT UI AND STATE MANAGEMENT ---

st.title("🚘 Number Plate Detection and Recognition")

# Initialize session state to store the detected plate number
if 'detected_plate' not in st.session_state:
    st.session_state.detected_plate = None

# Image capture from webcam
uploaded_img = st.camera_input("📷 Capture an image of a number plate")

# --- 3. IMAGE PROCESSING AND INFERENCE ---

if uploaded_img is not None:
    # Convert the uploaded image to an OpenCV format
    image_bytes = uploaded_img.getvalue()
    img_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    # The image from st.camera_input is already in BGR format, so no need to convert color channels
    
    original_shape = img_np.shape

    # Prepare image for YOLOv5
    img_resized = cv2.resize(img_np, (640, 640))
    # Note: YOLOv5 internally handles BGR to RGB conversion
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(yolo_model.device)

    # YOLOv5 Inference
    pred = yolo_model(img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.6) # Adjusted thresholds for potentially better accuracy

    detected_text_in_frame = None

    # Process detections
    for det in pred:
        if len(det):
            # Rescale boxes from img_size (640) to original image size
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], original_shape).round()
            
            # Process each detected box
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Add padding to the cropped plate for better OCR
                pad = 10 
                plate_crop = img_np[max(y1 - pad, 0):min(y2 + pad, original_shape[0]), max(x1 - pad, 0):min(x2 + pad, original_shape[1])]
                
                # Draw rectangle on the main image
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Use EasyOCR to read text from the cropped plate
                try:
                    # Upscale for better OCR results
                    plate_crop_upscaled = cv2.resize(plate_crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                    result = ocr_reader.readtext(plate_crop_upscaled)
                    
                    if result:
                        # Concatenate results and perform simple cleaning
                        plate_text = " ".join([r[-2] for r in result]).upper().replace(" ", "")
                        # Filter out common misreads like 'IND'
                        if "IND" not in plate_text and len(plate_text) > 4: # Basic sanity check
                            detected_text_in_frame = plate_text
                            break # Stop after finding the first valid plate
                except Exception as e:
                    st.warning(f"An error occurred during OCR: {e}")
            
            if detected_text_in_frame:
                break

    # Display results
    st.image(img_np, caption="Processed Image", channels="BGR", use_column_width=True)

    if detected_text_in_frame:
        st.success(f"✅ Detected Plate: **{detected_text_in_frame}**")
        if st.button("Confirm and Use This Plate"):
            st.session_state.detected_plate = detected_text_in_frame
            st.info(f"Plate '{st.session_state.detected_plate}' has been saved.")
            # st.experimental_rerun() # Optional: rerun to clear the image
    else:
        st.warning("⚠️ No valid number plate was detected in the image.")

# Display the stored plate number outside the main image processing block
st.sidebar.header("Stored Information")
if st.session_state.detected_plate:
    st.sidebar.success(f"Confirmed Plate: **{st.session_state.detected_plate}**")
    if st.sidebar.button("Clear Stored Plate"):
        st.session_state.detected_plate = None
        st.experimental_rerun()
else:
    st.sidebar.info("No plate has been confirmed yet.")
