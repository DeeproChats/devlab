# syntax=docker/dockerfile:1
FROM python:3.10-slim-bullseye


ENV DEBIAN_FRONTEND=noninteractive
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

# Install system deps (minimal + required for dlib, OpenCV, DeepFace)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    tesseract-ocr \
    poppler-utils \
    libtesseract-dev \
    pkg-config \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    sudo \
    ca-certificates \
    libopenblas-dev \
    liblapack-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user to match host UID/GID
RUN groupadd -g ${USER_GID} ${USERNAME} \
 && useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash ${USERNAME} \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
 && chmod 0440 /etc/sudoers.d/${USERNAME}

# Change working directory to the container root
WORKDIR /home/${USERNAME}

# Upgrade pip and install PyTorch CPU wheel first (special index)
RUN python -m pip install --upgrade pip
RUN python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install Python dependencies
COPY .devcontainer/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# Expose dev ports for Streamlit, Gradio, Jupyter
EXPOSE 5000 8501 7860 8888

# Switch to non-root user
USER ${USERNAME}

CMD ["bash"]
