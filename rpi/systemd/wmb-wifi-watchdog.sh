#!/bin/bash
set -e

# ----------------------------------------------------------------------
# WatchMyBirds WiFi Watchdog
# Checks if wlan0 has an IP. If not, reinstates AP mode.
# ----------------------------------------------------------------------

INTERFACE="wlan0"

# Check if we are already in AP mode (hostapd running)
if systemctl is-active --quiet hostapd; then
    echo "AP mode is already active. Watchdog sleeping."
    exit 0
fi

# Check if we have an IP address on wlan0
if ip addr show "$INTERFACE" | grep -q "inet "; then
    # We have an IP. Optionally, check internet connectivity.
    # For now, IP is enough proof of successful association.
    echo "WiFi connected. Everything is fine."
    exit 0
else
    echo "NO WiFi connection detected!"
    
    # Simple retry logic: force unblock and restart wpa_supplicant
    echo "Ensuring WiFi radio is unblocked..."
    rfkill unblock wifi || true
    
    echo "Attempting to restart wpa_supplicant..."
    systemctl restart wpa_supplicant@"$INTERFACE"
    sleep 20
    
    if ip addr show "$INTERFACE" | grep -q "inet "; then
         echo "Recovered after restart."
         exit 0
    fi

    # Still no connection? BRING BACK THE AP!
    echo "Connection failed. Reverting to AP Setup Mode."
    
    # Stop client services
    systemctl stop wpa_supplicant@"$INTERFACE"
    
    # Start AP services
    # We need to ensure the static IP for AP is set (handled by dnsmasq/hostapd often, 
    # but let's be safe and set it here as per AP_ABLAUF Phase 1)
    ip addr flush dev "$INTERFACE"
    ip addr add 192.168.4.1/24 dev "$INTERFACE"
    
    systemctl start dnsmasq
    systemctl start hostapd
    systemctl start wmb-setup-server
    
    echo "AP Mode restored. User can access http://192.168.4.1"
fi
