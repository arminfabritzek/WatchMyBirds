# Use a slim base image
FROM python:3.12-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libavcodec-extra \
    libxrender-dev \
    libavformat58 \
    libavfilter7 \
    libswscale5 \
    libavutil56 \
    libavdevice58 \
    libopenjp2-7 \
    ffmpeg \
    gosu \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade vulnerable libraries to fix CVEs
RUN apt-get update && \
    apt-get install --only-upgrade -y libxml2 libxslt1.1 && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/fontconfig \
    FONTCONFIG_PATH=/tmp/fontconfig \
    DEBUG_MODE=False \
    VIDEO_SOURCE=0 \
    OUTPUT_DIR=/output

# Set the working directory
WORKDIR /app

# OCI image labels
LABEL org.opencontainers.image.title="WatchMyBirds" \
      org.opencontainers.image.description="Bird detection and classification application" \
      org.opencontainers.image.source="https://github.com/arminfabritzek/WatchMyBirds" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}"

# Copy only requirements.txt first to leverage Docker cache for dependency installation
COPY requirements.txt /app/requirements.txt

# Install Python dependencies (upgrade pip, setuptools, and wheel first)
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code
COPY models ./models
COPY assets ./assets
COPY output ./output
COPY README.md ./
COPY utils ./utils
COPY logging_config.py ./
COPY main.py ./
COPY config.py ./
COPY camera ./camera
COPY detectors ./detectors
COPY web ./web

# Add the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose the port used by your app
EXPOSE 8050

# Use the entrypoint to dynamically handle users and permissions
ENTRYPOINT ["/entrypoint.sh"]

# Set the command to run your app
CMD ["waitress-serve", "--listen=0.0.0.0:8050", "main:app"]