#!/bin/bash

# Set default PUID and PGID if not provided
PUID=${PUID:-1000}
PGID=${PGID:-1000}
OUTPUT_DIR=${OUTPUT_DIR:-/output}
INGEST_DIR=${INGEST_DIR:-/ingest}
MODEL_BASE_PATH=${MODEL_BASE_PATH:-/models}

# Create a group with PGID if it doesn't exist
if ! getent group appgroup > /dev/null 2>&1; then
    groupadd -g $PGID appgroup
fi

# Create a user with PUID if it doesn't exist
if ! id -u appuser > /dev/null 2>&1; then
    useradd -u $PUID -g $PGID -m -s /bin/bash appuser 2>/dev/null || true
fi

# Ensure proper ownership and permissions for OUTPUT_DIR
mkdir -p "$OUTPUT_DIR"
chown -R $PUID:$PGID "$OUTPUT_DIR"
chown -R $PUID:$PGID /app/assets  # Ensure `appuser` owns the assets folder
mkdir -p "$MODEL_BASE_PATH"
chown -R $PUID:$PGID "$MODEL_BASE_PATH"  # Ensure `appuser` owns the models folder
mkdir -p /app/output
chown -R $PUID:$PGID /app/output  # Ensure `appuser` owns the models folder
mkdir -p "$INGEST_DIR"
chown -R $PUID:$PGID "$INGEST_DIR"

# Switch to the user and execute the CMD
exec gosu $PUID:$PGID "$@"
