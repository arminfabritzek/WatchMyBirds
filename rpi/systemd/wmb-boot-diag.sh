#!/bin/bash
# WatchMyBirds Boot Diagnostics
# Writes diagnostic info to /boot/firmware/debuglogs after every boot

set -e

BOOT_PARTITION="/boot/firmware"
DIAG_DIR="${BOOT_PARTITION}/debuglogs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CURRENT_DIR="${DIAG_DIR}/boot_${TIMESTAMP}"

# Ensure boot partition is mounted
if [ ! -d "$BOOT_PARTITION" ]; then
    echo "Boot partition not found at $BOOT_PARTITION"
    exit 1
fi

# Create diagnostics directory
mkdir -p "$CURRENT_DIR"

# Cleanup old diagnostics (keep last 3)
find "$DIAG_DIR" -mindepth 1 -maxdepth 1 -type d -name 'boot_*' | sort -r | tail -n +4 | xargs rm -rf 2>/dev/null || true

# Write summary
{
    echo "=== WatchMyBirds Boot Diagnostics ==="
    echo "Timestamp: $(date)"
    echo "Uptime: $(uptime)"
    echo ""
    echo "=== rfkill status ==="
    rfkill list 2>&1 || echo "rfkill not available"
    echo ""
    echo "=== Network Interfaces ==="
    ip addr 2>&1
    echo ""
    echo "=== WiFi Status ==="
    iw dev wlan0 link 2>&1 || echo "iw failed"
    echo ""
    echo "=== NetworkManager Device Status ==="
    nmcli device status 2>&1 || echo "nmcli not available"
    echo ""
    echo "=== Default Route ==="
    ip route 2>&1
} > "${CURRENT_DIR}/summary.txt"

# Network details
{
    echo "=== ip addr ==="
    ip addr
    echo ""
    echo "=== ip route ==="
    ip route
    echo ""
    echo "=== /etc/resolv.conf ==="
    cat /etc/resolv.conf 2>&1 || echo "No resolv.conf"
    echo ""
    echo "=== rfkill list ==="
    rfkill list 2>&1 || true
} > "${CURRENT_DIR}/network.txt"

# App status
{
    echo "=== App Service Status ==="
    systemctl status app.service 2>&1 || echo "App service not found"
} > "${CURRENT_DIR}/app.txt"

# System journal (last 200 lines)
journalctl -b --no-pager -n 200 > "${CURRENT_DIR}/system_journal.txt" 2>&1 || true

# App journal
journalctl -u app.service -b --no-pager -n 100 > "${CURRENT_DIR}/app_journal.txt" 2>&1 || true

# WPA supplicant journal
journalctl -u wpa_supplicant@wlan0 -b --no-pager -n 50 > "${CURRENT_DIR}/wpa_journal.txt" 2>&1 || true

# dhcpcd journal
journalctl -u dhcpcd -b --no-pager -n 50 > "${CURRENT_DIR}/dhcpcd_journal.txt" 2>&1 || true

# Watchdog journal
journalctl -u wmb-wifi-watchdog.service -b --no-pager -n 50 > "${CURRENT_DIR}/watchdog_journal.txt" 2>&1 || true

echo "Boot diagnostics written to ${CURRENT_DIR}"
