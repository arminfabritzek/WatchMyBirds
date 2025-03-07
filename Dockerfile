# Use a slim base image
FROM python:3.12-bullseye

# Set the build argument for the timestamp
ARG BUILD_TIMESTAMP

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libavcodec-extra \
    libxrender-dev \
    ffmpeg \
    gosu \
    fontconfig \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/fontconfig \
    FONTCONFIG_PATH=/tmp/fontconfig \
    DEBUG_MODE=False \
    VIDEO_SOURCE=0 \
    OUTPUT_DIR=/output

# Set the working directory
WORKDIR /app

# Copy only requirements.txt first to leverage Docker cache for dependency installation
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code
COPY . /app

# Add the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Write the build timestamp to a file
RUN echo "Build timestamp: ${BUILD_TIMESTAMP}" > /app/build_timestamp.txt

# Expose the port used by your app
EXPOSE 8050

# Use the entrypoint to dynamically handle users and permissions
ENTRYPOINT ["/entrypoint.sh"]

# Set the command to run your app
CMD ["waitress-serve", "--listen=0.0.0.0:8050", "main:server"]