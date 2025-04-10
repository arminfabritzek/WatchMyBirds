# Use a slim base image
FROM python:3.12-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libavcodec-extra \
    libxrender-dev \
    ffmpeg \
    gosu \
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