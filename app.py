import sys
import os
import torch
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Add yolov5 folder to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLOV5_PATH = os.path.join(BASE_DIR, 'yolov5')
sys.path.append(YOLOV5_PATH)

# Import from YOLOv5 modules after appending to sys.path
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox


# Load YOLOv5 model
model_path = os.path.join(YOLOV5_PATH, 'runs', 'train', 'exp2', 'weights', 'best_linux.pt')
model = attempt_load(model_path, map_location='cpu')
model.eval()
# device = select_device('cpu')  # or 'cuda:0' if GPU is available
# model_path = 'yolov5/runs/train/exp/weights/best.pt'  # or wherever your file is
# Correct relative path to the model file
# model_path = os.path.join("car-recognition-streamlit","yolov5", "runs", "train", "exp2", "weights", "best.pt")

# Title
st.title("🚘 Number Plate Detection")

# Capture from camera
uploaded_img = st.camera_input("📷 Capture Image")

if uploaded_img is not None:
    # Convert to OpenCV format
    image = Image.open(uploaded_img)
    img = np.array(image.convert("RGB"))
    original_shape = img.shape

    # YOLO input preparation
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = img_resized[:, :, ::-1].copy().transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img_rgb).float().div(255.0).unsqueeze(0).to(device)

    # Inference
    pred = model(img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.65)

    detected_text = None

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], original_shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                pad = 20
                x1 = max(x1 - pad, 0)
                y1 = max(y1 - pad, 0)
                x2 = min(x2 + pad, img.shape[1])
                y2 = min(y2 + pad, img.shape[0])

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                plate_crop = img[y1:y2, x1:x2]
                plate_crop = cv2.resize(plate_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                result = reader.readtext(plate_crop)
                if result:
                    plate_text = " ".join([r[-2] for r in result])
                    if "IND" not in plate_text:
                        detected_text = plate_text
                        break

    # Display
    st.image(img, caption="🔍 Detected Number Plate", channels="BGR", use_column_width=True)

    if detected_text:
        st.success(f"✅ Detected Plate: {detected_text}")
        if st.button("✅ Use This Plate"):
            st.info(f"Plate '{detected_text}' accepted.")
        if st.button("❌ Clear"):
            st.experimental_rerun()
    else:
        st.warning("⚠️ No valid plate detected (or 'IND' filtered).")
