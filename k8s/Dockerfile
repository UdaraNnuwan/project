# Use the slim Python image to keep the container small.
FROM python:3.9-slim

# Stop Python from writing .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1
# Send logs to Kubernetes immediately.
ENV PYTHONUNBUFFERED=1

# Set the app folder.
WORKDIR /app

# Install system packages here if a Python package needs build tools.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Copy requirements first so Docker can cache dependency installs.
COPY k8s/requirements.txt .

# Install Python dependencies without saving pip cache.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the inference script, model code, model weights, and helpers.
COPY k8s/live_inference.py .
COPY alibaba_trace/model_architecture.py .
COPY alibaba_trace/outputs/dual_head_model.pt .
COPY alibaba_trace/utils/notifications.py ./alibaba_trace/utils/

# Start the live inference script.
CMD ["python", "live_inference.py"]
