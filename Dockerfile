# ─────────────────────────────────────────────
# Base image with CUDA 12.9
# ─────────────────────────────────────────────
FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04

# Avoid prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# ─────────────────────────────────────────────
# Install system dependencies
# ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set python alias
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# ─────────────────────────────────────────────
# Set working directory
# ─────────────────────────────────────────────
WORKDIR /app

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install Python dependencies (excluding torch)
RUN pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# Install PyTorch CUDA 12.9
# ─────────────────────────────────────────────
RUN pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu129

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
