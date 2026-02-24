# Raspberry Pi Appliance Guide

This directory contains utility scripts, first-boot logic, and configuration for the WatchMyBirds Raspberry Pi Appliance.

## 1. Directory Structure

- **`ap/`**: Configuration for the Access Point fallback (Hostapd/Dnsmasq).
- **`first-boot/`**: Critical initialization scripts (`first-boot.sh`) executed once on a fresh image.
- **`systemd/`**: System service definitions (`app.service`, `app-dev.service`) and helper units.
- **`harden.sh` / `harden-dev.sh`**: Build-time scripts executed inside the image to apply security policies.

---

## 2. First Boot Setup (Headless)

When you flash the image and boot the Raspberry Pi for the first time, it performs the following steps:

1. **Filesystem Expansion:** Resizes the root partition to fill the SD card.
2. **WiFi Configuration Check:**
   - Checks if you provided WiFi credentials (via Imager or `wpa_supplicant.conf`).
   - Checks if AP Mode was explicitly requested via a `wmb-ap` file in boot.
3. **Network Decision:**
   - **Scenario A (WiFi Configured):** Attempts to connect to your WiFi.
   - **Scenario B (No WiFi):** Enters **AP Mode** automatically.

### Access Point (AP) Mode
If the device cannot connect to WiFi after 60 seconds (or has no config), it creates its own network:
- **SSID:** `WatchMyBirds-XXXX` (XXXX = Serial Number Suffix)
- **Password:** `watchmybirds`
- **Management IP:** `192.168.4.1`

**Setup Flow:**
1. Connect your phone/laptop to the AP WiFi.
2. Open `http://192.168.4.1:8050/setup`.
3. Select your home network and enter the password.
4. The device saves the config and reboots into **Client Mode**.

---

## 3. Development vs. Production Images

We produce two types of images:

| Feature | Golden Image (Production) | Dev Image (Development) |
| :--- | :--- | :--- |
| **Service** | `app.service` (Hardened) | `app-dev.service` (Relaxed) |
| **SSH** | Disabled | **Enabled** |
| **Admin User**| Locked | **Unlocked (passwordless sudo)** |
| **Filesystem**| Read-Only (`ProtectSystem=strict`) | Writable `/opt/app` |

> **Warning:** Never expose a Dev Image to the internet. It is intended for local debugging only.

---

## 4. Manual Maintenance

### Enabling SSH (Production)
SSH is disabled by default. To enable it:
1. Shutdown the Pi and put the SD card in your computer.
2. Create an empty file named `ssh` (no extension) in the `bootfs` partition.
3. Boot the Pi. SSH will be enabled.

### Viewing Logs
The system logs to `systemd-journald`.
```bash
# View app logs
sudo journalctl -u app -f

# View go2rtc logs
sudo journalctl -u go2rtc -f

# View boot initialization logs
cat /boot/firmware/first-boot.log
```

### Streaming Runtime
- The appliance image installs and enables `go2rtc.service` by default.
- On first boot (client mode), `go2rtc` is started before `app.service`.
- The app uses `STREAM_SOURCE_MODE=auto` behavior:
  - go2rtc healthy -> relay mode
  - go2rtc unavailable -> direct fallback
