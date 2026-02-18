#!/bin/bash
set -euo pipefail

# ----------------------------------------------------------------------
# WatchMyBirds WiFi Watchdog
# - Keeps client mode stable across short router/network outages.
# - Reverts to AP mode only after repeated recovery failures.
# ----------------------------------------------------------------------

INTERFACE="wlan0"
STATE_DIR="/var/lib/watchmybirds"
FAIL_COUNTER_FILE="$STATE_DIR/wifi_watchdog_failures"
MAX_CONSECUTIVE_FAILURES="${WMB_WD_MAX_FAILURES:-5}"
RECOVERY_WAIT_SEC="${WMB_WD_RECOVERY_WAIT_SEC:-30}"
BOOT_GRACE_SEC="${WMB_WD_BOOT_GRACE_SEC:-600}"

mkdir -p "$STATE_DIR"

has_ipv4() {
    ip -4 addr show "$INTERFACE" | grep -q "inet "
}

read_failures() {
    if [ -f "$FAIL_COUNTER_FILE" ]; then
        local raw
        raw="$(cat "$FAIL_COUNTER_FILE" 2>/dev/null || echo 0)"
        if [[ "$raw" =~ ^[0-9]+$ ]]; then
            echo "$raw"
        else
            echo 0
        fi
    else
        echo 0
    fi
}

reset_failures() {
    echo 0 > "$FAIL_COUNTER_FILE"
}

increment_failures() {
    local current
    current="$(read_failures)"
    current=$((current + 1))
    echo "$current" > "$FAIL_COUNTER_FILE"
    echo "$current"
}

recover_client_mode_once() {
    echo "NO WiFi IP on $INTERFACE. Attempting recovery..."
    rfkill unblock wifi || true
    systemctl restart wpa_supplicant@"$INTERFACE" || true
    systemctl restart dhcpcd || true
    sleep "$RECOVERY_WAIT_SEC"
}

# Already in AP mode: nothing to do.
if systemctl is-active --quiet hostapd; then
    echo "AP mode is already active. Watchdog sleeping."
    exit 0
fi

# Healthy client mode: reset failure counter.
if has_ipv4; then
    reset_failures
    echo "WiFi connected. Counter reset."
    exit 0
fi

uptime_sec="$(cut -d. -f1 /proc/uptime 2>/dev/null || echo 0)"
if [ "$uptime_sec" -lt "$BOOT_GRACE_SEC" ]; then
    echo "Within boot grace (${uptime_sec}s < ${BOOT_GRACE_SEC}s). Deferring AP fallback."
    recover_client_mode_once
    if has_ipv4; then
        reset_failures
        echo "Recovered during boot grace."
        exit 0
    fi
    failures="$(increment_failures)"
    echo "Still no IP during boot grace (failures=$failures)."
    exit 0
fi

recover_client_mode_once
if has_ipv4; then
    reset_failures
    echo "Recovered after restart."
    exit 0
fi

failures="$(increment_failures)"
if [ "$failures" -lt "$MAX_CONSECUTIVE_FAILURES" ]; then
    echo "Recovery failed (failures=$failures/$MAX_CONSECUTIVE_FAILURES). Retrying on next timer run."
    exit 0
fi

echo "Recovery failed repeatedly (failures=$failures). Reverting to AP Setup Mode."

# Stop client services
systemctl stop wpa_supplicant@"$INTERFACE" || true
systemctl stop dhcpcd || true

# Start AP services
ip addr flush dev "$INTERFACE" || true
ip addr add 192.168.4.1/24 dev "$INTERFACE"
systemctl start dnsmasq
systemctl start hostapd
systemctl start wmb-setup-server

reset_failures
echo "AP Mode restored. User can access http://192.168.4.1"
