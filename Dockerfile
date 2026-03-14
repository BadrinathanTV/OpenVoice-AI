# 1. Base Image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 2. Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

# 3. Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    curl \
    git \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Map python3 to python3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3

# Install Node.js 18.x
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install uv globally
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/root/.local/bin" sh

# 4. Set working directory
WORKDIR /app

# 5. Copy project files
COPY . .

# 6. Setup Backend
WORKDIR /app
# Install Python dependencies from the root project metadata
RUN uv sync

# Backend imports live under /app/backend/src
WORKDIR /app/backend

# Pre-download models by running a short python script
RUN uv run --project /app python -c "\
import os;\
from src.tts.piper import TTSModel;\
print('Downloading TTS Models...');\
tts1 = TTSModel('en_US-bryce-medium');\
tts2 = TTSModel('en_GB-alba-medium');\
tts3 = TTSModel('en_US-hfc_female-medium');\
print('Done downloading models.');\
"

# 7. Setup Frontend
WORKDIR /app/frontend
RUN npm install
RUN npm run build

# 8. Set Entrypoint
WORKDIR /app
COPY entrypoint.sh ./entrypoint.sh
# Make entrypoint executable (just in case)
RUN chmod +x entrypoint.sh

# Expose ports
EXPOSE 8000
EXPOSE 5173

# Start services
CMD ["./entrypoint.sh"]
