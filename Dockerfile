# ---------- Stage 1: Build analytics ----------
FROM --platform=$BUILDPLATFORM node:20-bookworm AS analytics-build

WORKDIR /analytics
COPY analytics/package.json analytics/package-lock.json ./
RUN npm ci

COPY analytics/ .
RUN npm run build


# ---------- Stage 2: Python runtime ----------
FROM python:3.12-slim-bookworm

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libopenjp2-7 \
    gosu && \
    apt-get autoremove -y && \
    apt-get autoclean -y && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV XDG_CACHE_HOME=/tmp/fontconfig \
    FONTCONFIG_PATH=/tmp/fontconfig \
    DEBUG_MODE=False \
    VIDEO_SOURCE=0 \
    OUTPUT_DIR=/output \
    INGEST_DIR=/ingest \
    MODEL_BASE_PATH=/models

# Set the working directory
WORKDIR /app

ARG GIT_COMMIT
ARG BUILD_DATE
ARG VERSION

# OCI image labels
LABEL org.opencontainers.image.title="WatchMyBirds" \
    org.opencontainers.image.description="Bird detection and classification application" \
    org.opencontainers.image.source="https://github.com/arminfabritzek/WatchMyBirds" \
    org.opencontainers.image.version="${VERSION}" \
    org.opencontainers.image.revision="${GIT_COMMIT}" \
    org.opencontainers.image.created="${BUILD_DATE}"

# Going for CPU-only for now
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.9.1 torchvision==0.24.1

COPY requirements.txt /app/requirements.txt

# Install Python dependencies (upgrade pip, setuptools, and wheel first)
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code
COPY analytics ./analytics
COPY assets ./assets
COPY camera ./camera
COPY detectors ./detectors
COPY templates ./templates
COPY utils ./utils
COPY web ./web
COPY config.py ./
COPY logging_config.py ./
COPY main.py ./
COPY README.md ./

# Copy analytics build
COPY --from=analytics-build /assets/analytics/ /app/assets/analytics/

# Create runtime directories (no model/output copy at build time)
RUN mkdir -p /models /output /ingest

# Add the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose the port used by your app
EXPOSE 8050

# Use the entrypoint to dynamically handle users and permissions
ENTRYPOINT ["/entrypoint.sh"]

# Set the command to run your app
CMD ["waitress-serve", "--listen=0.0.0.0:8050", "main:app"]
