# Use a Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies, including build tools
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    git \
    build-essential \
    cmake \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt for caching purposes
COPY requirements.txt ./

# Install dependencies one-by-one to pinpoint the error
RUN pip install --no-cache-dir streamlit
RUN pip install --no-cache-dir opencv-python
RUN pip install --no-cache-dir easyocr
RUN pip install --no-cache-dir Pillow
RUN pip install --no-cache-dir torch
RUN pip install --no-cache-dir git+https://github.com/ultralytics/yolov5.git

# Copy the app code and other files
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8501

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py"]
