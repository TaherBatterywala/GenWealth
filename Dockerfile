# ==============================================================================
# GenWealth AI — Production Multi-Platform Dockerfile
# Compatible with: Hugging Face Spaces (Free 16GB RAM), Render, and Local Docker
# ==============================================================================

FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860

# Install minimal OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install CPU-only PyTorch first to keep Docker image lightweight and fast
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy dependency definition
COPY requirements.txt .

# Install dependencies (torch is already installed as CPU version)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, and frontend
COPY app/ /app/app/
COPY src/ /app/src/
COPY model_artifacts/ /app/model_artifacts/
COPY frontend/ /app/frontend/

# Create a non-root user for security (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user && \
    chown -R user:user /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose common container ports (7860 for Hugging Face, 8000 for standard)
EXPOSE 7860 8000

# Start Uvicorn using the dynamic PORT environment variable (supports HF 7860, Render 10000, and local)
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
