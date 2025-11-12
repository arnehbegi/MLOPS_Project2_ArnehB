# Use Python 3.10 slim image for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch CPU version and other dependencies
# Using CPU-only PyTorch to reduce image size significantly
RUN pip install --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY model.py .
COPY data_module.py .

# Create directory for checkpoints
RUN mkdir -p /app/models /app/hf_cache
# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command - runs training with best hyperparameters from Project 1
# Users can override these by passing arguments to docker run
CMD ["python", "main.py", \
     "--checkpoint_dir", "/app/models", \
     "--lr", "5.5e-5", \
     "--weight_decay", "0.001", \
     "--warmup_steps", "0", \
     "--no_wandb"]