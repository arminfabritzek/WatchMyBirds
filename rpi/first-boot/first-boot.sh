#!/bin/bash
set -x

# ----------------------------------------------------------------------
# WatchMyBirds First Boot Setup
# ----------------------------------------------------------------------

# LOGGING TO BOOT PARTITION
# Ensures users can debug headless boot failures by reading the SD card on a PC.
LOGFILE="/boot/firmware/first-boot.log"

# Fallback for older RaspiOS layouts if /boot/firmware isn't the mount
if [ ! -d "/boot/firmware" ] && [ -d "/boot" ]; then
    LOGFILE="/boot/first-boot.log"
fi

# Rotate old log if it exists (Keep one previous run)
if [ -f "$LOGFILE" ]; then
    mv "$LOGFILE" "${LOGFILE}.old"
fi

# Redirect stdout/stderr to logfile
exec > >(tee -a "${LOGFILE}") 2>&1

echo "================================================================================"
echo "WatchMyBirds First Boot Log - Started at $(date)"
echo "================================================================================"

# Debug: List boot partition contents to verify trigger file presence
echo "DEBUG: Listing Boot Partition Contents:"
ls -la /boot/firmware/ || true
ls -la /boot/ || true

# 0. First-Boot Gate (repeatable via trigger)
# ----------------------------------------------------------------------
MARKER="/var/lib/watchmybirds/.first-boot-done"
TRIGGER_BOOT="/boot/firmware/wmb-first-boot"
TRIGGER_BOOT_LEGACY="/boot/wmb-first-boot"

TRIGGER_PRESENT=0
if [ -f "$TRIGGER_BOOT" ] || [ -f "$TRIGGER_BOOT_LEGACY" ]; then
    TRIGGER_PRESENT=1
fi

if [ -f "$MARKER" ] && [ "$TRIGGER_PRESENT" -eq 0 ]; then
    echo "First-boot already completed; no trigger found. Exiting."
    exit 0
fi

FORCE_AP=0
if [ "$TRIGGER_PRESENT" -eq 1 ]; then
    echo "First-boot trigger found. Forcing AP setup mode."
    FORCE_AP=1
fi

mkdir -p "$(dirname "$MARKER")"

# 0. Ingest WiFi Config (Headless Support)
# ----------------------------------------------------------------------
# If user provided wpa_supplicant.conf in boot partition, install it.
if [ -f /boot/firmware/wpa_supplicant.conf ]; then
    echo "Found wpa_supplicant.conf in /boot/firmware. Moving to /etc/..."
    mv /boot/firmware/wpa_supplicant.conf /etc/wpa_supplicant/wpa_supplicant.conf
    chmod 600 /etc/wpa_supplicant/wpa_supplicant.conf
    chown root:root /etc/wpa_supplicant/wpa_supplicant.conf
fi

# 0b. Ingest SSH Identity
# ----------------------------------------------------------------------
if [ -f /boot/firmware/authorized_keys ]; then
    echo "Found authorized_keys. Installing for user 'admin'..."
    mkdir -p /home/admin/.ssh
    mv /boot/firmware/authorized_keys /home/admin/.ssh/authorized_keys
    chown -R admin:admin /home/admin/.ssh
    chmod 700 /home/admin/.ssh
    chmod 600 /home/admin/.ssh/authorized_keys
fi

# 1. SSH Handling (Standard RPi Mechanism)
# ----------------------------------------------------------------------
if [ -f /boot/firmware/ssh ] || [ -f /boot/ssh ]; then
    echo "SSH Marker found. Enabling SSH..."
    ssh-keygen -A
    systemctl enable --now ssh
    rm -f /boot/firmware/ssh /boot/ssh
else
    echo "SSH Marker not found. SSH remains disabled."
fi

# 2. Network / AP Mode
# ----------------------------------------------------------------------
# Strict Validation for NetworkManager
# Checks for [wifi] section containing non-empty ssid AND [wifi-security] or 802-1x presence
HAS_NM=0
for f in /etc/NetworkManager/system-connections/*.nmconnection; do
    [ -e "$f" ] || continue
    # Check for SSID assignment (ssid=...) and Security/Auth section
    if grep -qE "^ssid=.+$" "$f" && grep -qE "\[wifi-security\]|key-mgmt=wpa-eap" "$f"; then
        HAS_NM=1
        break
    fi
done

# Strict Validation for wpa_supplicant
# Checks for non-empty ssid AND (psk or key_mgmt other than NONE implies auth)
# Simple heuristic: Does it look like a configured network block?
HAS_WPA=0
if grep -qE "ssid=\"[^\"]+\"" /etc/wpa_supplicant/wpa_supplicant.conf && \
   grep -qE "psk=|key_mgmt=WPA|key_mgmt=SAE" /etc/wpa_supplicant/wpa_supplicant.conf; then
    HAS_WPA=1
fi

IS_AP=0

if [ "$FORCE_AP" -eq 1 ]; then
    echo "Trigger present. Forcing AP Mode for setup."
    IS_AP=1
elif [ "$HAS_NM" -gt 0 ] || [ "$HAS_WPA" -eq 1 ]; then
    echo "WiFi Configuration found ($HAS_NM NM, $HAS_WPA WPA). Booting in Client Mode."
    
    # ------------------------------------------------------------------
    # CLIENT MODE SETUP
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # CLIENT MODE SETUP
    # ------------------------------------------------------------------
    echo "Ensuring Client Mode services are active..."
    
    # SAFETY: Stop AP services to prevent race conditions
    systemctl disable --now hostapd 2>/dev/null || true
    systemctl disable --now dnsmasq 2>/dev/null || true
    systemctl disable --now wmb-setup-server.service 2>/dev/null || true
    
    # Flush IP to ensure clean slate
    ip addr flush dev wlan0 || true

    # Ensure NM doesn't interfere if wpa_supplicant is used
    if [ "$HAS_WPA" -eq 1 ] && command -v nmcli >/dev/null; then
        nmcli dev set wlan0 managed no 2>/dev/null || true
    fi
    
    # Ensure WiFi interface is up
    rfkill unblock wlan 2>/dev/null || true
    ip link set wlan0 up 2>/dev/null || true
    
    if [ "$HAS_WPA" -eq 1 ]; then
        echo "Starting wpa_supplicant..."
        # Ensure country code presence (critical for RPi WiFi)
        if ! grep -q "country=" /etc/wpa_supplicant/wpa_supplicant.conf; then
            sed -i '1i country=DE' /etc/wpa_supplicant/wpa_supplicant.conf
        fi
        
        # Creating interface-specific config for wpa_supplicant@wlan0
        cp /etc/wpa_supplicant/wpa_supplicant.conf /etc/wpa_supplicant/wpa_supplicant-wlan0.conf
        
        # Start specific interface service to ensure binding
        echo "Starting wpa_supplicant@wlan0..."
        systemctl enable --now wpa_supplicant@wlan0
        
        # Ensure DHCP client is running on wlan0
        echo "Starting DHCP client..."
        systemctl unmask dhcpcd 2>/dev/null || true
        systemctl enable --now dhcpcd 2>/dev/null || true
    fi
    
    # ------------------------------------------------------------------
    # VALIDATION LOOP (The Fence)
    # ------------------------------------------------------------------
    echo "Waiting for WiFi Connection (Max 60s)..."
    CONNECTED=0
    for i in $(seq 1 30); do
        # Robust check: Link connected AND IP assigned
        LINK_STATUS=$(iw dev wlan0 link)
        IP_STATUS=$(ip -4 addr show wlan0 | grep "inet ")
        
        if [[ "$LINK_STATUS" == *"Connected to"* ]] && [[ -n "$IP_STATUS" ]]; then
            echo "SUCCESS: WiFi connected with valid IP."
            CONNECTED=1
            break
        fi
        sleep 2
    done
    
    if [ "$CONNECTED" -eq 1 ]; then
        # --------------------------------------------------------------
        # SUCCESS PATH
        # --------------------------------------------------------------
        echo "Enabling WatchMyBirds application..."
        systemctl enable app.service 2>/dev/null || true
        systemctl start app.service || true
        
        # Diagnostics for App Start
        sleep 5
        if ! systemctl is-active --quiet app.service; then
            echo "ERROR: App failed to start! Dumping logs to bootfs..."
            journalctl -u app.service -n 50 --no-pager
        fi
        
    else
        # --------------------------------------------------------------
        # FAILURE PATH (Fallback to AP)
        # --------------------------------------------------------------
        echo "FAILURE: Could not obtain IP address after 60s."
        
        # DEBUG: Dump logs before killing services
        DEBUG_DIR="/boot/firmware/debuglogs/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$DEBUG_DIR"
        echo "--- WRITING DEBUG LOGS TO $DEBUG_DIR ---"
        
        journalctl -b -u wpa_supplicant@wlan0 --no-pager -n 200 > "$DEBUG_DIR/wpa_supplicant.log" 2>&1 || true
        systemctl cat wpa_supplicant@wlan0.service > "$DEBUG_DIR/wpa_supplicant_service_def.txt" 2>&1 || true
        journalctl -b -u dhcpcd --no-pager -n 200 > "$DEBUG_DIR/dhcpcd.log" 2>&1 || true
        
        {
            echo "=== IP ADDR ==="
            ip addr
            echo "=== IP ROUTE ==="
            ip route
            echo "=== RFKILL ==="
            rfkill list
            echo "=== IW LINK ==="
            iw dev wlan0 link
        } > "$DEBUG_DIR/network_state.txt" 2>&1

        echo "Reverting to Access Point Mode..."
        
        # 1. Kill Client Services
        systemctl stop wpa_supplicant@wlan0 || true
        systemctl stop wpa_supplicant || true
        systemctl stop dhcpcd || true
        
        # 2. Force AP Flag for this session to jump to AP logic
        
        SUFFIX=$(grep "Serial" /proc/cpuinfo | awk '{print $3}' | tail -c 5)
        if [ -z "$SUFFIX" ]; then SUFFIX="FAIL"; fi
        sed -i "s/WatchMyBirds-XXXX/WatchMyBirds-$SUFFIX/" /etc/hostapd/hostapd.conf
        
        ip link set wlan0 down
        ip addr flush dev wlan0
        ip link set wlan0 up
        ip addr add 192.168.4.1/24 dev wlan0
        
        systemctl enable --now hostapd
        systemctl enable --now dnsmasq
        systemctl enable --now wmb-setup-server.service
        
        echo "Fallback Complete. Device is now an AP (WatchMyBirds-$SUFFIX)."
        IS_AP=1
    fi
else
    echo "No WiFi Configuration. Booting in AP Mode."
    IS_AP=1
    
    # Generate SSID Suffix from Serial (last 4 chars)
    SUFFIX=$(grep "Serial" /proc/cpuinfo | awk '{print $3}' | tail -c 5)
    if [ -z "$SUFFIX" ]; then SUFFIX="INIT"; fi
    
    sed -i "s/WatchMyBirds-XXXX/WatchMyBirds-$SUFFIX/" /etc/hostapd/hostapd.conf
    
    # Prevent NM from managing wlan0 (Avoid conflicts)
    if command -v nmcli >/dev/null; then
        nmcli dev set wlan0 managed no || true
    fi
    
    # Ensure WiFi is unblocked and interface is up (required on newer RPi OS)
    rfkill unblock wlan || true
    ip link set wlan0 down || true
    ip addr flush dev wlan0 || true
    ip link set wlan0 up

    # Scan for nearby SSIDs (best effort) to prefill setup UI
    SSID_SCAN_FILE="/var/lib/watchmybirds/ssid_scan.txt"
    if command -v iw >/dev/null; then
        iw dev wlan0 scan 2>/dev/null | \
            awk -F: '/SSID:/{sub(/^ /,"",$2); if($2!="") print $2}' | \
            sort -u > "$SSID_SCAN_FILE" || true
    elif command -v iwlist >/dev/null; then
        iwlist wlan0 scan 2>/dev/null | \
            awk -F: '/ESSID:/{gsub(/"/,"",$2); if($2!="") print $2}' | \
            sort -u > "$SSID_SCAN_FILE" || true
    else
        : > "$SSID_SCAN_FILE"
    fi
    chmod 644 "$SSID_SCAN_FILE" || true

    # Safety: Stop wpa_supplicant to avoid conflicts
    systemctl stop wpa_supplicant || true
    # Safety: Stop dhcpcd to prevent it from fighting over the static AP IP
    systemctl stop dhcpcd || true
    systemctl stop wmb-setup-server.service 2>/dev/null || true

    # Set Static IP for AP Gateway (Non-persistent, strictly for first-run)
    ip addr add 192.168.4.1/24 dev wlan0
    
    systemctl enable --now hostapd
    systemctl enable --now dnsmasq
    systemctl enable --now wmb-setup-server.service || true
    
    # ------------------------------------------------------------------
    # DIAGNOSTICS (Wait for startup and log status)
    # ------------------------------------------------------------------
    echo "DEBUG: Waiting 5s for services to stabilize..."
    sleep 5
    
    echo "DEBUG: Network Status (ip link / ip addr):"
    ip link show wlan0 || true
    ip addr show wlan0 || true
    
    echo "DEBUG: RFKill Status:"
    rfkill list || true
    
    echo "DEBUG: Hostapd Status:"
    systemctl status hostapd --no-pager || true
    if ! systemctl is-active --quiet hostapd; then
        echo "DEBUG: HOSTAPD FAILED! Dumping logs:"
        journalctl -u hostapd --no-pager -n 50 || true
    fi
    
    echo "DEBUG: Dnsmasq Status:"
    systemctl status dnsmasq --no-pager || true
    if ! systemctl is-active --quiet dnsmasq; then
        echo "DEBUG: DNSMASQ FAILED! Dumping logs:"
        journalctl -u dnsmasq --no-pager -n 50 || true
    fi
fi

# 3. Security / UFW
# ----------------------------------------------------------------------
echo "Enforcing Security..."
# Reset to be safe
ufw --force reset
ufw default deny incoming
ufw default allow outgoing

# App / Setup UI
if [ "$IS_AP" -eq 1 ]; then
    echo "Allowing AP Services on wlan0..."
    ufw allow in on wlan0 to any port 80 proto tcp comment 'Setup UI'
    ufw allow in on wlan0 to any port 67 proto udp
    ufw allow in on wlan0 to any port 53
else
    ufw allow 8050/tcp comment 'Web Interface'
fi

# Allow SSH if service is ENABLED (starts on boot) - prevents race condition vs is-active
if systemctl is-enabled --quiet ssh; then
    echo "SSH Service enabled. Allowing Port 22..."
    ufw allow 22/tcp comment 'SSH (explicitly enabled)'
else
    # Ensure port is closed only if service is disabled
    ufw delete allow 22/tcp 2>/dev/null || true
fi

# Enable
ufw --force enable

# 4. Cleanup & Self-Disable
# ----------------------------------------------------------------------
echo "Marking First-Boot Complete..."
touch "$MARKER"
rm -f "$TRIGGER_BOOT" "$TRIGGER_BOOT_LEGACY"

# Snapshot State / STATUS.txt (Item #1)
STATUS_FILE="/boot/firmware/STATUS.txt"
if [ ! -d "/boot/firmware" ] && [ -d "/boot" ]; then
    STATUS_FILE="/boot/STATUS.txt"
fi

{
    echo "TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')"
    if [ "$IS_AP" -eq 1 ]; then
        echo "MODE=AP"
        SUFFIX=$(grep "Serial" /proc/cpuinfo | awk '{print $3}' | tail -c 5)
        [ -z "$SUFFIX" ] && SUFFIX="XXXX"
        echo "SSID=WatchMyBirds-$SUFFIX"
    else
        echo "MODE=CLIENT"
        CURRENT_SSID=$(iw dev wlan0 link | grep SSID | awk '{print $2}')
        echo "SSID=$CURRENT_SSID"
    fi
    echo "IW_STATUS=$(iw dev wlan0 link | grep 'Connected' || echo 'Not Connected')"
    echo "IPV4=$(ip -4 addr show wlan0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}')"
    # Fallback grep for systems without -P
    if [ -z "$IPV4" ]; then
         echo "IPV4=$(ip -4 addr show wlan0 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)"
    fi
    echo "DEFAULT_ROUTE=$(ip route | grep default | awk '{print $3}')"
    
    echo "--- APP STATUS ---"
    systemctl status app.service || true
    echo "--- WIFI STATUS ---"
    systemctl status hostapd || true
    systemctl status wpa_supplicant@wlan0 || true
    systemctl status wpa_supplicant || true

} > "$STATUS_FILE"

# ----------------------------------------------------------------------
# ENHANCED DIAGNOSTICS (Headless Verification)
# ----------------------------------------------------------------------
# Always dump App Status/Logs to boot partition for inspection
DIAG_DIR="/boot/firmware/debuglogs/startup_$(date +%Y%m%d_%H%M%S)"
# Fallback for older RaspiOS layouts or if firmware folder is missing
if [ ! -d "/boot/firmware" ] && [ -d "/boot" ]; then
    DIAG_DIR="/boot/debuglogs/startup_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$DIAG_DIR"

echo "Waiting 15s for services to stabilize..."
sleep 15

# Cleanup old logs (Keep last 48h).
CLEANUP_ROOT="$(dirname "$DIAG_DIR")"
if [[ "$CLEANUP_ROOT" == *"/debuglogs" ]]; then
    find "$CLEANUP_ROOT" -mindepth 1 -maxdepth 1 -type d -name "startup_*" -mtime +2 -exec rm -rf {} + 2>/dev/null || true
fi

echo "--- Network Status ---" > "$DIAG_DIR/network.txt"
ip addr >> "$DIAG_DIR/network.txt" 2>&1
ip route >> "$DIAG_DIR/network.txt" 2>&1
iwconfig >> "$DIAG_DIR/network.txt" 2>&1
cat /etc/resolv.conf >> "$DIAG_DIR/network.txt" 2>&1

echo "--- App Service Status ---" > "$DIAG_DIR/app_status.txt"
systemctl is-active app.service >> "$DIAG_DIR/app_status.txt" 2>&1
systemctl status app.service >> "$DIAG_DIR/app_status.txt" 2>&1

echo "--- Critical Systemd Properties ---" > "$DIAG_DIR/systemd_props.txt"
systemctl show app.service -p StateDirectory,ReadWritePaths,ProtectSystem,WorkingDirectory >> "$DIAG_DIR/systemd_props.txt"

echo "--- Directory Permissions ---" > "$DIAG_DIR/dir_perms.txt"
ls -ld /var/lib/watchmybirds >> "$DIAG_DIR/dir_perms.txt" 2>&1
ls -la /var/lib/watchmybirds >> "$DIAG_DIR/dir_perms.txt" 2>&1
ls -ld /opt/app >> "$DIAG_DIR/dir_perms.txt" 2>&1

echo "--- Kernel Messages (dmesg) ---" > "$DIAG_DIR/dmesg.log"
dmesg | tail -n 500 >> "$DIAG_DIR/dmesg.log" 2>&1

echo "--- Tool Verification ---" > "$DIAG_DIR/tools.txt"
ffmpeg -version >> "$DIAG_DIR/tools.txt" 2>&1 || echo "ffmpeg MISSING" >> "$DIAG_DIR/tools.txt"
v4l2-ctl --version >> "$DIAG_DIR/tools.txt" 2>&1 || echo "v4l-utils MISSING" >> "$DIAG_DIR/tools.txt"

journalctl -u app.service -b --no-pager -n 200 > "$DIAG_DIR/app_journal.log" 2>&1
journalctl -b --no-pager -n 200 > "$DIAG_DIR/sys_journal.log" 2>&1  # Full system log snippet

echo "Diagnostics complete. Logs written to $DIAG_DIR"
