import streamlit as st
import torch
from pathlib import Path
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.datasets import LoadImages
from yolov5.utils.torch_utils import select_device
import cv2
import numpy as np

# Set page config
st.set_page_config(page_title="Vehicle Number Plate Detection", layout="centered")

# Title
st.title("🚘 Vehicle Number Plate Detection using YOLOv5")

# Load model (safe load)
@st.cache_resource
def load_model():
    model_path = Path("best.pt")
    device = select_device('cpu')
    model = DetectMultiBackend(str(model_path), device=device)
    model.eval()
    return model

model = load_model()

# Image upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # YOLOv5 detection
    with st.spinner("Running detection..."):
        img_resized = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        for det in pred:
            if len(det):
                for *xyxy, conf, cls in det:
                    label = f"{model.names[int(cls)]} {conf:.2f}"
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        st.image(img, caption="Detection Result", use_column_width=True)
