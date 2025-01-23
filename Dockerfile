# Use a base image compatible with your application
FROM python:3.11-slim-bookworm


# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    iputils-ping \
    telnet \
    wget \
    tar \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV XDG_CACHE_HOME=/tmp/fontconfig

# Create directories for the model and app
RUN mkdir -p /home/appuser/.cache/torch/hub/checkpoints/ /tmp/fontconfig/torch/hub/checkpoints/
WORKDIR /app

# Copy only necessary files (see .dockerignore for exclusions)
COPY . /app

# Download the pre-trained model file and copy it to the desired locations
RUN wget https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth -O /tmp/ssd300_vgg16_coco-b556d3b4.pth && \
    cp /tmp/ssd300_vgg16_coco-b556d3b4.pth /home/appuser/.cache/torch/hub/checkpoints/ && \
    cp /tmp/ssd300_vgg16_coco-b556d3b4.pth /tmp/fontconfig/torch/hub/checkpoints/ && \
    rm /tmp/ssd300_vgg16_coco-b556d3b4.pth

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Add a non-root user with customizable UID/GID
ARG PUID=1000
ARG PGID=1000
RUN groupadd -g $PGID appgroup && useradd -u $PUID -g $PGID -ms /bin/bash appuser

# Create the output directory and set ownership before switching users
RUN mkdir -p /output && chown appuser:appgroup /output
RUN mkdir -p /tmp/matplotlib /tmp/fontconfig && chown -R appuser:appgroup /tmp/matplotlib /tmp/fontconfig

# Ensure all files are readable by all users
RUN chmod -R a+r /app/assets

# Switch to non-root user
USER appuser

# Expose the port used by your app
EXPOSE 5001

# Define environment variables with default values (can be overridden)
ENV VIDEO_SOURCE=0 OUTPUT_DIR=/output

# Set the command to run your app
CMD ["waitress-serve", "--listen=0.0.0.0:5001", "main:app"]