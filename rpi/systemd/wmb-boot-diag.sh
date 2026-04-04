#!/bin/bash
# WatchMyBirds Boot Diagnostics
# Writes diagnostic info to /boot/firmware/debuglogs after every boot

set -e

BOOT_PARTITION="/boot/firmware"
if [ ! -d "$BOOT_PARTITION" ] && [ -d "/boot" ]; then
    BOOT_PARTITION="/boot"
fi

DIAG_DIR="${BOOT_PARTITION}/debuglogs"
STATUS_FILE="${BOOT_PARTITION}/STATUS.txt"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CURRENT_DIR="${DIAG_DIR}/boot_${TIMESTAMP}"

# Ensure boot partition is mounted
if [ ! -d "$BOOT_PARTITION" ]; then
    echo "Boot partition not found at $BOOT_PARTITION"
    exit 1
fi

get_wlan_ipv4() {
    ip -4 addr show wlan0 2>/dev/null | awk '/inet / {print $2}' | cut -d/ -f1 | head -n1
}

read_watchdog_failures() {
    local failure_file="/var/lib/watchmybirds/wifi_watchdog_failures"
    if [ -f "$failure_file" ]; then
        cat "$failure_file" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

write_status_snapshot() {
    local iw_status
    local ipv4
    local default_route
    local mode="CLIENT"
    local ssid=""

    if systemctl is-active --quiet hostapd; then
        mode="AP"
        ssid="WatchMyBirds-$(grep "Serial" /proc/cpuinfo | awk '{print $3}' | tail -c 5)"
        [ "$ssid" = "WatchMyBirds-" ] && ssid="WatchMyBirds-XXXX"
    else
        ssid="$(iw dev wlan0 link 2>/dev/null | awk '/SSID/ {print $2; exit}')"
    fi

    iw_status="$(iw dev wlan0 link 2>/dev/null | grep 'Connected' || echo 'Not Connected')"
    ipv4="$(get_wlan_ipv4)"
    default_route="$(ip route 2>/dev/null | awk '/^default/ {print $3; exit}')"

    {
        echo "TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')"
        echo "SOURCE=wmb-boot-diag"
        echo "BOOT_DIAG_DIR=$(basename "$CURRENT_DIR")"
        echo "MODE=$mode"
        echo "SSID=$ssid"
        echo "IW_STATUS=$iw_status"
        echo "IPV4=$ipv4"
        echo "DEFAULT_ROUTE=$default_route"
        echo "APP_ACTIVE=$(systemctl is-active app.service 2>/dev/null || true)"
        echo "APP_ENABLED=$(systemctl is-enabled app.service 2>/dev/null || true)"
        echo "APP_RESULT=$(systemctl show app.service -p Result --value 2>/dev/null || true)"
        echo "APP_MAINPID=$(systemctl show app.service -p MainPID --value 2>/dev/null || true)"
        echo "FIRST_BOOT_ACTIVE=$(systemctl is-active wmb-first-boot.service 2>/dev/null || true)"
        echo "FIRST_BOOT_RESULT=$(systemctl show wmb-first-boot.service -p Result --value 2>/dev/null || true)"
        echo "HOSTAPD_ACTIVE=$(systemctl is-active hostapd 2>/dev/null || true)"
        echo "WPA_WLAN0_ACTIVE=$(systemctl is-active wpa_supplicant@wlan0 2>/dev/null || true)"
        echo "WIFI_WATCHDOG_SERVICE=$(systemctl is-active wmb-wifi-watchdog.service 2>/dev/null || true)"
        echo "WIFI_WATCHDOG_TIMER=$(systemctl is-active wmb-wifi-watchdog.timer 2>/dev/null || true)"
        echo "WATCHDOG_FAILURES=$(read_watchdog_failures)"
        echo "PENDING_WIFI_CONFIG=$([ -f /opt/app/data/pending_wifi.conf ] && echo yes || echo no)"
        echo "PENDING_ADMIN_PASSWORD=$([ -f /opt/app/data/pending_admin_password ] && echo yes || echo no)"
        echo "--- APP STATUS ---"
        systemctl status app.service --no-pager || true
        echo "--- FIRST BOOT STATUS ---"
        systemctl status wmb-first-boot.service --no-pager || true
        echo "--- WIFI STATUS ---"
        systemctl status hostapd --no-pager || true
        systemctl status wpa_supplicant@wlan0 --no-pager || true
    } > "$STATUS_FILE"
}

# Create diagnostics directory
mkdir -p "$CURRENT_DIR"
write_status_snapshot

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
    systemctl show app.service -p ActiveState -p SubState -p Result -p UnitFileState -p MainPID -p ExecMainStatus -p ExecMainCode 2>&1 || true
    echo ""
    systemctl status app.service --no-pager 2>&1 || true
} > "${CURRENT_DIR}/app.txt"

# First-boot status
{
    echo "=== First-Boot Service Status ==="
    systemctl show wmb-first-boot.service -p ActiveState -p SubState -p Result -p MainPID -p ExecMainStatus -p ExecMainCode 2>&1 || true
    echo ""
    systemctl status wmb-first-boot.service --no-pager 2>&1 || true
} > "${CURRENT_DIR}/first_boot.txt"

# System journal (last 200 lines)
journalctl -b --no-pager -n 200 > "${CURRENT_DIR}/system_journal.txt" 2>&1 || true

# App journal
journalctl -u app.service -b --no-pager -n 100 > "${CURRENT_DIR}/app_journal.txt" 2>&1 || true

# First-boot journal
journalctl -u wmb-first-boot.service -b --no-pager -n 200 > "${CURRENT_DIR}/first_boot_journal.txt" 2>&1 || true

# WPA supplicant journal
journalctl -u wpa_supplicant@wlan0 -b --no-pager -n 50 > "${CURRENT_DIR}/wpa_journal.txt" 2>&1 || true

# dhcpcd journal
journalctl -u dhcpcd -b --no-pager -n 50 > "${CURRENT_DIR}/dhcpcd_journal.txt" 2>&1 || true

# Watchdog journal
journalctl -u wmb-wifi-watchdog.service -b --no-pager -n 50 > "${CURRENT_DIR}/watchdog_journal.txt" 2>&1 || true

echo "Boot diagnostics written to ${CURRENT_DIR}"
