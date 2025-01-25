# Use a base image compatible with your application
FROM python:3.12-slim

# Set the build argument for the timestamp
ARG BUILD_TIMESTAMP

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libx264-dev \
    libavcodec-extra \
    libxrender-dev \
    ffmpeg \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    wget \
    gosu \
    fontconfig \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV XDG_CACHE_HOME=/tmp/fontconfig
ENV FONTCONFIG_PATH=/tmp/fontconfig
ENV DEBUG_MODE=False

WORKDIR /app

# Copy the application files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the pre-trained model file and copy it to the desired locations
RUN mkdir -p /tmp/fontconfig/torch/hub/checkpoints/ && \
    wget https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth -O /tmp/fontconfig/torch/hub/checkpoints/ssd300_vgg16_coco-b556d3b4.pth

# Add the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Write the build timestamp to a file
RUN echo "Build timestamp: ${BUILD_TIMESTAMP}" > /app/build_timestamp.txt

# Expose the port used by your app
EXPOSE 5001

# Set environment variables
ENV VIDEO_SOURCE=0 OUTPUT_DIR=/output

# Use the entrypoint to dynamically handle users and permissions
ENTRYPOINT ["/entrypoint.sh"]

# Set the command to run your app
CMD ["waitress-serve", "--listen=0.0.0.0:5001", "main:app"]