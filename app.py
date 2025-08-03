import streamlit as st
import torch
import cv2
import os
import numpy as np
from PIL import Image
import easyocr
from pathlib import Path
import sys

# Set YOLOv5 Path
BASE_DIR = os.getcwd()
st.title(BASE_DIR)
YOLO_PATH = os.path.join(BASE_DIR, 'yolov5')
MODEL_PATH = os.path.join(YOLO_PATH,"runs","train","exp2", "weights", "best.pt")

# Append YOLOv5 path to sys.path
sys.path.append(str(YOLO_PATH))

# Import YOLOv5 internal modules
from yolov5.utils.general import scale_boxes, non_max_suppression
from yolov5.utils.torch_utils import select_device
from yolov5.models.common import DetectMultiBackend

# Streamlit Title
st.title("🔍 Number Plate Detection using YOLOv5 and EasyOCR")

# Load YOLOv5 model
@st.cache_resource
def load_yolo_model():
    device = select_device('cpu')
    model = DetectMultiBackend(str(MODEL_PATH), device=device)
    model.eval()
    return model

# Load EasyOCR Reader
@st.cache_resource
def load_easyocr_reader():
    return easyocr.Reader(['en'])

# Process the uploaded image
def process_image(uploaded_image, yolo_model, reader):
    image = np.array(uploaded_image.convert("RGB"))
    img_resized = cv2.resize(image, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    pred = yolo_model(img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    annotated_img = image.copy()
    plates = []

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], image.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                plate_crop = image[y1:y2, x1:x2]

                if plate_crop.size == 0:
                    continue

                try:
                    result = reader.readtext(plate_crop)
                    text = result[0][1] if result else 'N/A'
                    plates.append((text, conf.item()))

                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_img, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                except Exception as e:
                    st.warning(f"OCR error: {e}")

    return annotated_img, plates

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with st.spinner("Loading YOLOv5 model and OCR engine..."):
        yolo_model = load_yolo_model()
        reader = load_easyocr_reader()

    image = Image.open(uploaded_file)
    st.image(image, caption='📸 Uploaded Image', use_column_width=True)

    with st.spinner("🔎 Detecting and reading number plates..."):
        annotated_img, plates = process_image(image, yolo_model, reader)

    st.image(annotated_img, caption='✅ Detected Plates', use_column_width=True)

    if plates:
        st.markdown("### 🧾 Extracted Number Plates")
        for i, (text, conf) in enumerate(plates, 1):
            st.write(f"**Plate {i}**: `{text}` (Confidence: `{conf:.2f}`)")
    else:
        st.info("No number plates detected.")
