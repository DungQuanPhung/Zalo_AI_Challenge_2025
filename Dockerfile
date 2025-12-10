# Dockerfile for Zalo AI Challenge 2025 - AeroEyes
# Base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy model files from save_models folder
COPY save_models/yolo.engine ./yolo.engine
COPY save_models/yolo.onnx ./yolo.onnx
COPY save_models/reid.onnx ./reid.onnx

# Copy inference script
COPY main.py .

# Create result directory
RUN mkdir -p /result

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command - run inference on /data and save to /result/submission.json
CMD ["python", "main.py", "--model", "yolo.engine", "--data", "/data", "--output", "/result/submission.json", "--conf", "0.5"]