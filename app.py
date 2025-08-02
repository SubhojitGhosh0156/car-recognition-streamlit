import sys
import os
import torch
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import easyocr

# 👇 Add yolov5 path
YOLO_PATH = os.path.join(os.getcwd(), 'yolov5')
sys.path.append(YOLO_PATH)

from utils.general import scale_boxes, non_max_suppression
from utils.torch_utils import select_device
from models.common import DetectMultiBackend

# 📌 Model setup
device = select_device('cpu')
reader = easyocr.Reader(['en'], gpu=False)
model_path = os.path.join(os.getcwd(), 'yolov5', 'runs', 'train', 'exp2', 'weights', 'best.pt')


# ✅ Load model
model = torch.load(model_path, map_location=device)['model'].float()
model.eval()


# 🖼️ Streamlit UI
st.title("🚘 Number Plate Detection")

uploaded_img = st.camera_input("📷 Capture Image from Webcam")

if uploaded_img is not None:
    image = Image.open(uploaded_img)
    img = np.array(image.convert("RGB"))
    original_shape = img.shape

    img_resized = cv2.resize(img, (640, 640))
    img_rgb = img_resized[:, :, ::-1].copy().transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img_rgb).float().div(255.0).unsqueeze(0).to(device)

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
                    if "IND" not in plate_text.upper() and "INO" not in plate_text.upper():
                        detected_text = plate_text

    st.image(img, caption="Detected Number Plate", channels="BGR", use_column_width=True)

    if detected_text:
        st.success(f"✅ Detected Plate: {detected_text}")
        if st.button("✅ Use This Plate"):
            st.info(f"Plate '{detected_text}' accepted.")
        if st.button("❌ Clear"):
            st.experimental_rerun()
    else:
        st.warning("⚠️ No valid plate detected (or 'IND' filtered).")
