# ---------- Python runtime ----------
# Pinned to multi-arch index digest so amd64 and arm64 builds stay reproducible.
# Dependabot keeps this up-to-date; always use the index (multi-arch) digest,
# never a platform-specific one, or arm64 builds will break.
FROM python:3.14-slim-bookworm@sha256:980c03657c7c8bfbce5212d242ffe5caf69bfd8b6c8383e3580b27d028a6ddb3

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

COPY requirements.txt /app/requirements.txt
COPY requirements-aesthetic.txt /app/requirements-aesthetic.txt

# Install Python dependencies (upgrade pip, setuptools, and wheel first)
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Install the aesthetic-tagger stack (torch CPU-only, open_clip).
# Kept as a separate pip invocation so the index pin doesn't interfere
# with the main resolver run. Adds ~900 MB to the image; if you want a
# slim variant without the tagger, comment this RUN out and the
# in-app scheduler will skip itself when the imports fail.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r requirements-aesthetic.txt

# Copy the rest of the application source code
COPY assets ./assets
COPY camera ./camera
COPY core ./core
COPY detectors ./detectors
COPY scripts ./scripts
COPY templates ./templates
COPY utils ./utils
COPY web ./web
COPY config.py ./
COPY logging_config.py ./
COPY main.py ./
COPY README.md ./
COPY go2rtc.yaml.example ./

# Copy analytics build
# Create runtime directories (no model/output copy at build time)
RUN mkdir -p /models /output /ingest

# Persist build metadata as runtime-readable files. Use ${VAR:-unknown}
# fallbacks so that an empty build-arg never produces an empty file --
# read_build_metadata() reports "Unknown" only for missing/empty files,
# so an explicit "unknown" token at least signals "build forgot the arg".
RUN echo "${VERSION:-unknown}" > /app/APP_VERSION && \
    echo "${GIT_COMMIT:-unknown}" > /app/BUILD_COMMIT && \
    echo "${BUILD_DATE:-unknown}" > /app/BUILD_DATE

# Add the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose the port used by your app
EXPOSE 8050

# Set runtime path defaults to match Docker mount convention
ENV OUTPUT_DIR="/output"
ENV INGEST_DIR="/ingest"
ENV MODEL_BASE_PATH="/models"

# Use entrypoint script to handle user permissions
ENTRYPOINT ["/entrypoint.sh"]

# Set the command to run your app
CMD ["python", "main.py"]
