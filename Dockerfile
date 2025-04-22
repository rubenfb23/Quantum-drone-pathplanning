FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get install -y python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src

# Set working directory and default command
WORKDIR /app/src
RUN nvidia-smi
CMD ["python3", "main.py"]