# ==========================================
# RECOVERED DOCKERFILE FOR auto_annotate_ext
# ==========================================
# This Dockerfile has been reverse-engineered from your existing Docker image layers.

FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# Performance and library config paths
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
ENV MPLCONFIGDIR=/tmp/matplotlib

# Install system dependencies (including GUI and database drivers originally present)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    nano \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libx11-xcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    libxcb-cursor0 \
    libegl1 \
    libopengl0 \
    unixodbc-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyQt6 for optional GUI components
RUN pip install --no-cache-dir PyQt6

# Install standard core dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    websockets \
    pyodbc \
    pymssql \
    pymodbus \
    pandas \
    "numpy<2.0" \
    requests \
    supervision \
    opencv-python

# Install Ultralytics YOLOv8 & YOLO-World
RUN pip install --no-cache-dir ultralytics

# Install Facebook Segment Anything Model (SAM) from source
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git

# Install Grounding DINO with CUDA support from source
RUN pip install --no-cache-dir --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git

# Prepare weight storage and environment variables
RUN mkdir -p /opt/weights
ENV MODEL_WEIGHTS=/opt/weights/yolov8s-seg.pt

# Install additional utility dependencies originally present
RUN pip install --no-cache-dir \
    sqlalchemy \
    imageio \
    python-multipart \
    "python-jose[cryptography]" \
    "passlib[bcrypt]" \
    "git+https://github.com/ultralytics/CLIP.git"

# Set up application workspace
WORKDIR /app

# Expose FastAPI port
EXPOSE 8000

# Default command
CMD ["bash"]
