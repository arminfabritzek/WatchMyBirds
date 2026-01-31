#!/bin/bash
set -e

# ----------------------------------------------------------------------
# WatchMyBirds WiFi Setup Helper
# Triggered by systemd path unit when /var/lib/watchmybirds/pending_wifi.conf appears
# ----------------------------------------------------------------------

PENDING_FILE="/var/lib/watchmybirds/pending_wifi.conf"
TARGET_CONF="/etc/wpa_supplicant/wpa_supplicant.conf"

if [ ! -f "$PENDING_FILE" ]; then
    echo "No pending configuration found."
    exit 0
fi

echo "Reading pending WiFi configuration..."
# Read SSID and PSK carefully (handling spaces)
# Format is expected to be simple JSON or Key-Value. 
# Let's assume the web app writes a simple line-based format or we parse it here.
# Simpler: The web app writes the ready-to-use wpa_supplicant network block?
# NO, parsing user input in bash is risky. 
# Better: Web app writes JSON, we use python to generate the file?
# OR: Web app writes the standard wpa_supplicant.conf content directly?
# Since the web app validates inputs, letting it write the full file content is acceptable 
# IF we trust the web app input validation. The file is owned by the app user.

# Move the file to boot
echo "Installing WiFi configuration to SD Card..."
mv "$PENDING_FILE" "$TARGET_CONF"
chown root:root "$TARGET_CONF"
chmod 600 "$TARGET_CONF"

# Remove the internal state marker so next boot runs clean
rm -f "/var/lib/watchmybirds/.first-boot-done"

echo "Disabling AP and setup services..."
systemctl disable --now wmb-setup-server.service || true
systemctl disable --now hostapd || true
systemctl disable --now dnsmasq || true

echo "Enabling WiFi Watchdog..."
systemctl enable --now wmb-wifi-watchdog.timer || true

echo "Rebooting system to apply changes..."
reboot
