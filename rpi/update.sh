#!/bin/bash
# ------------------------------------------------------------------------------
# rpi/update.sh — WatchMyBirds OTA Updater
# ------------------------------------------------------------------------------
# Triggered by wmb-update.service (runs as root).
#
# Reads the requested target version from:
#   /opt/app/data/update_request.txt
#
# Writes status updates to:
#   /opt/app/data/update_status.json
#
# Valid targets:
#   "main"        → installs latest commit from the main branch
#   "v0.2.0"      → installs the tarball from the matching GitHub release tag
# ------------------------------------------------------------------------------

set -euo pipefail

APP_DIR="/opt/app"
DATA_DIR="${APP_DIR}/data"
REQUEST_FILE="${DATA_DIR}/update_request.txt"
STATUS_FILE="${DATA_DIR}/update_status.json"
TMP_DIR="/tmp/wmb-update-$$"
BACKUP_DIR="${DATA_DIR}/backups"
GITHUB_REPO="arminfabritzek/WatchMyBirds"
APP_SERVICE="app.service"
APP_USER="watchmybirds"

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

now_iso() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

write_status() {
    local state="$1"
    local message="$2"
    local target="${3:-}"
    printf '{\n  "state": "%s",\n  "message": "%s",\n  "target": "%s",\n  "timestamp": "%s"\n}\n' \
        "$state" "$message" "$target" "$(now_iso)" > "$STATUS_FILE"
    chown "${APP_USER}:${APP_USER}" "$STATUS_FILE" 2>/dev/null || true
    echo "[wmb-update] [$state] $message"
}

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# ------------------------------------------------------------------------------
# Validate request file
# ------------------------------------------------------------------------------

if [ ! -f "$REQUEST_FILE" ]; then
    echo "[wmb-update] No update_request.txt found. Exiting."
    exit 0
fi

TARGET=$(cat "$REQUEST_FILE" | tr -d '[:space:]')

# Basic allowlist: only alphanumeric, dots, dashes, underscores, forward slash
if ! echo "$TARGET" | grep -qE '^[a-zA-Z0-9._/-]{1,64}$'; then
    write_status "error" "Invalid target: ${TARGET}" "$TARGET"
    exit 1
fi

echo "[wmb-update] Starting update. Target: $TARGET"
write_status "downloading" "Preparing to download ${TARGET}..." "$TARGET"

# ------------------------------------------------------------------------------
# Determine download URL
# ------------------------------------------------------------------------------

mkdir -p "$TMP_DIR"

if [ "$TARGET" = "main" ]; then
    DOWNLOAD_URL="https://github.com/${GITHUB_REPO}/archive/refs/heads/main.tar.gz"
    STRIP_DIR="WatchMyBirds-main"
else
    # Normalise: strip leading 'v' for comparison but keep tag as-is for URL
    TAG="$TARGET"
    DOWNLOAD_URL="https://github.com/${GITHUB_REPO}/archive/refs/tags/${TAG}.tar.gz"
    # GitHub strips the leading 'v' in the extracted directory name
    STRIP_DIR="WatchMyBirds-${TAG#v}"
fi

# ------------------------------------------------------------------------------
# Download
# ------------------------------------------------------------------------------

TARBALL="${TMP_DIR}/update.tar.gz"
echo "[wmb-update] Downloading from: $DOWNLOAD_URL"
write_status "downloading" "Downloading ${TARGET}..." "$TARGET"

if ! curl -fsSL --max-time 120 --retry 3 --retry-delay 5 \
        -o "$TARBALL" "$DOWNLOAD_URL"; then
    write_status "error" "Download failed for ${TARGET}." "$TARGET"
    exit 1
fi

echo "[wmb-update] Download complete ($(du -sh "$TARBALL" | cut -f1))."

# ------------------------------------------------------------------------------
# Extract
# ------------------------------------------------------------------------------

write_status "installing" "Extracting ${TARGET}..." "$TARGET"
tar -xzf "$TARBALL" -C "$TMP_DIR"

EXTRACTED_DIR="${TMP_DIR}/${STRIP_DIR}"
if [ ! -d "$EXTRACTED_DIR" ]; then
    # Fallback: find the single extracted directory
    EXTRACTED_DIR=$(find "$TMP_DIR" -mindepth 1 -maxdepth 1 -type d | head -n1)
fi

if [ -z "$EXTRACTED_DIR" ] || [ ! -d "$EXTRACTED_DIR" ]; then
    write_status "error" "Could not locate extracted source directory." "$TARGET"
    exit 1
fi

echo "[wmb-update] Extracted to: $EXTRACTED_DIR"

# ------------------------------------------------------------------------------
# Backup current installation (keep last 3 backups)
# ------------------------------------------------------------------------------

write_status "installing" "Backing up current installation..." "$TARGET"
BACKUP_STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_STAMP}"
mkdir -p "$BACKUP_DIR"

# Copy app code (exclude data directory and virtual environment — too large)
rsync -a --exclude='data/' --exclude='.venv/' --exclude='__pycache__/' \
    "${APP_DIR}/" "${BACKUP_PATH}/" 2>/dev/null || true

# Prune old backups — keep the 3 most recent
ls -1dt "${BACKUP_DIR}"/20* 2>/dev/null | tail -n +4 | xargs rm -rf 2>/dev/null || true

echo "[wmb-update] Backup created at: $BACKUP_PATH"

# ------------------------------------------------------------------------------
# Stop the app service
# ------------------------------------------------------------------------------

write_status "installing" "Stopping app service..." "$TARGET"
systemctl stop "$APP_SERVICE" || true
sleep 2

# ------------------------------------------------------------------------------
# Install new files
# ------------------------------------------------------------------------------

write_status "installing" "Installing new files..." "$TARGET"

# Sync all source files; skip the data directory and virtualenv
rsync -a --delete \
    --exclude='data/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    "${EXTRACTED_DIR}/" "${APP_DIR}/"

# Fix ownership
chown -R "${APP_USER}:${APP_USER}" "${APP_DIR}" 2>/dev/null || true
# Ensure data directory stays owned by app user
chown "${APP_USER}:${APP_USER}" "${DATA_DIR}" 2>/dev/null || true

echo "[wmb-update] Files installed."

# ------------------------------------------------------------------------------
# Update Python dependencies
# ------------------------------------------------------------------------------

VENV="${APP_DIR}/.venv"
REQUIREMENTS="${APP_DIR}/requirements.txt"

if [ -f "$REQUIREMENTS" ] && [ -d "$VENV" ]; then
    write_status "installing" "Updating Python dependencies..." "$TARGET"
    "${VENV}/bin/pip" install --quiet --no-cache-dir -r "$REQUIREMENTS" \
        || { write_status "error" "pip install failed." "$TARGET"; exit 1; }
    echo "[wmb-update] Dependencies updated."
fi

# ------------------------------------------------------------------------------
# Remove stale .pyc files
# ------------------------------------------------------------------------------

find "${APP_DIR}" -name '*.pyc' -delete 2>/dev/null || true
find "${APP_DIR}" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# ------------------------------------------------------------------------------
# Restart the app service
# ------------------------------------------------------------------------------

write_status "restarting" "Restarting app service..." "$TARGET"
systemctl start "$APP_SERVICE"

# Wait a moment and verify it came up
sleep 5
if systemctl is-active --quiet "$APP_SERVICE"; then
    echo "[wmb-update] App service is running."
    write_status "success" "Updated to ${TARGET} successfully." "$TARGET"
else
    echo "[wmb-update] WARNING: App service did not start cleanly after update."
    write_status "error" "Update applied but app failed to start. Check logs." "$TARGET"
    # Don't exit 1 here — the files were installed; operator can investigate
fi

# Clean up request file
rm -f "$REQUEST_FILE"

echo "[wmb-update] Done."
