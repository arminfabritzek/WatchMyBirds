#!/bin/bash
set -e

# ----------------------------------------------------------------------
# WatchMyBirds WiFi Setup Helper
# Triggered by systemd path unit when /opt/app/data/pending_wifi.conf appears
# ----------------------------------------------------------------------

PENDING_FILE="/opt/app/data/pending_wifi.conf"
PENDING_PASSWORD_FILE="/opt/app/data/pending_admin_password"
TARGET_CONF="/etc/wpa_supplicant/wpa_supplicant.conf"
SETTINGS_PATH="/opt/app/data/output/settings.yaml"

if [ ! -f "$PENDING_FILE" ]; then
    echo "No pending configuration found."
    exit 0
fi

echo "Reading pending WiFi configuration..."
# The web app writes the full wpa_supplicant.conf content directly and
# validates all user input before doing so. The file is owned by the app
# user, so bash does not need to reparse or reformat the contents.

# Move the file to boot
echo "Installing WiFi configuration to SD Card..."
mv "$PENDING_FILE" "$TARGET_CONF"
chown root:root "$TARGET_CONF"
chmod 600 "$TARGET_CONF"

if [ -f "$PENDING_PASSWORD_FILE" ]; then
    echo "Installing admin password into runtime settings..."
    export WMB_PENDING_PASSWORD_FILE="$PENDING_PASSWORD_FILE"
    export WMB_SETTINGS_PATH="$SETTINGS_PATH"
    /opt/app/.venv/bin/python - <<'PY'
import os
from pathlib import Path

import yaml

pending_path = Path(os.environ["WMB_PENDING_PASSWORD_FILE"])
settings_path = Path(os.environ["WMB_SETTINGS_PATH"])
password = pending_path.read_text(encoding="utf-8").strip()

if not password:
    raise SystemExit("Pending admin password is empty.")

settings_path.parent.mkdir(parents=True, exist_ok=True)
settings = {}
if settings_path.exists():
    raw = settings_path.read_text(encoding="utf-8").strip()
    if raw:
        loaded = yaml.safe_load(raw) or {}
        if isinstance(loaded, dict):
            settings = loaded

settings["EDIT_PASSWORD"] = password

tmp_path = settings_path.with_suffix(".tmp")
with tmp_path.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(settings, handle, sort_keys=True)

os.chmod(tmp_path, 0o600)
tmp_path.replace(settings_path)
PY
    chown watchmybirds:watchmybirds "$SETTINGS_PATH"
    chmod 600 "$SETTINGS_PATH"
    rm -f "$PENDING_PASSWORD_FILE"
fi

# Remove the internal state marker so next boot runs clean
rm -f "/opt/app/data/.first-boot-done"

echo "Disabling AP and setup services..."
systemctl disable --now wmb-setup-server.service || true
systemctl disable --now hostapd || true
systemctl disable --now dnsmasq || true

echo "Enabling WiFi Watchdog..."
systemctl enable --now wmb-wifi-watchdog.timer || true

echo "Rebooting system to apply changes..."
reboot
