import streamlit as st
import torch
import cv2
import os
import numpy as np
from PIL import Image
import easyocr
import sys
import seaborn as sn
# Paths
BASE_DIR = os.getcwd()
YOLO_PATH = os.path.join(BASE_DIR, "yolov5")
sys.path.append(YOLO_PATH)

from yolov5.utils.general import scale_boxes, non_max_suppression
from yolov5.utils.torch_utils import select_device
from yolov5.models.common import DetectMultiBackend

# Load YOLOv5 model
@st.cache_resource
def load_yolo_model():
    model_path = os.path.join(YOLO_PATH, 'runs', 'train', 'exp2', 'weights', 'best.pt')
    device = select_device('cpu')
    model = DetectMultiBackend(model_path, device=device)
    return model

# Load EasyOCR reader
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)

# Streamlit UI
st.title("🚘 Number Plate Recognition")
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    yolo_model = load_yolo_model()
    ocr_reader = load_ocr_reader()

    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    original_shape = img_np.shape

    # Resize and prepare tensor
    img_resized = cv2.resize(img_np, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # Inference
    pred = yolo_model(img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.45)

    plates = []

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], original_shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                plate_crop = img_np[y1:y2, x1:x2]

                if plate_crop.size == 0:
                    continue

                plate_crop = cv2.resize(plate_crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

                try:
                    result = ocr_reader.readtext(plate_crop)
                    if result:
                        plate_text = result[0][1]
                        plates.append(plate_text)
                        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img_np, plate_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                except Exception as e:
                    st.warning(f"OCR Error: {e}")

    st.image(img_np, caption="🔍 Detected Plates", channels="BGR", use_column_width=True)

    if plates:
        st.markdown("### ✅ Extracted Number Plates")
        for i, text in enumerate(plates, 1):
            st.write(f"**Plate {i}**: `{text}`")
    else:
        st.info("No number plates detected.")
