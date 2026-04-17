# We use the 'slim' variant of the python Debian image for a much smaller footprint
FROM python:3.9-slim

# Prevent Python from writing .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
# Disables stdout/stderr buffering, providing immediate log output to the K8s container logs
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies (if you need build tools for certain ML pip packages, uncomment)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Copy requirements separately to leverage Docker layer caching
COPY k8s/requirements.txt .

# Install Python requirements cleanly and securely
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all the model codes and inference script
COPY k8s/live_inference.py .
COPY alibaba_trace/model_architecture.py .
COPY alibaba_trace/outputs/dual_head_model.pt .
COPY src/notifications.py .

# Ensure the script runs natively
CMD ["python", "live_inference.py"]
