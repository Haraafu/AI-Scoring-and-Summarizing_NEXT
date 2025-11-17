# Base image Python
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential wget curl \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY ./src ./src

# Expose port (required for Cloud Run)
EXPOSE 8080

# Start FastAPI server
CMD ["uvicorn", "src.service_batch:app", "--host", "0.0.0.0", "--port", "8080"]
