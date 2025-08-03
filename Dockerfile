# Use a Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies, including build tools and git
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

# Clone the yolov5 repository directly
RUN git clone https://github.com/ultralytics/yolov5.git

# Copy requirements.txt and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code and other files
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8501

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py"]
