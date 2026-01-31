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

# Ensure proper ownership for mounted volumes (if they exist)
# Directories are created by App if missing, but Chown handles permissions for Docker mounts
[ -d "$OUTPUT_DIR" ] && chown -R $PUID:$PGID "$OUTPUT_DIR"
chown -R $PUID:$PGID /app/assets || true
[ -d "$MODEL_BASE_PATH" ] && chown -R $PUID:$PGID "$MODEL_BASE_PATH"
[ -d "$INGEST_DIR" ] && chown -R $PUID:$PGID "$INGEST_DIR"

# Switch to the user and execute the CMD
exec gosu $PUID:$PGID "$@"
