#!/bin/bash
set -e
set -x

# ----------------------------------------------------------------------
# WatchMyBirds Golden Image Hardening Script
# executed inside QEMU chroot
# ----------------------------------------------------------------------

export DEBIAN_FRONTEND=noninteractive
export APT_LISTCHANGES_FRONTEND=none

# Optional: Disable update-initramfs to speed up and prevent prompts
echo "update_initramfs=no" >> /etc/initramfs-tools/update-initramfs.conf || true

# ----------------------------------------------------------------------
# 0. Logging Persistence (CRITICAL for debugging)
# ----------------------------------------------------------------------
echo "Configuring Persistent Logging..."
mkdir -p /var/log/journal
# Use --root to avoid requiring /proc during image build chroot.
systemd-tmpfiles --root=/ --create --prefix /var/log/journal
# Force persistence in journald.conf
mkdir -p /etc/systemd/journald.conf.d
printf "[Journal]\nStorage=persistent\nSystemMaxUse=500M\n" > /etc/systemd/journald.conf.d/persistence.conf

# Set Timezone (Default to Berlin for German deployment)
echo "Europe/Berlin" > /etc/timezone
ln -sf /usr/share/zoneinfo/Europe/Berlin /etc/localtime

# 1. Update & Install Dependencies
# ----------------------------------------------------------------------
apt-get update

# Define non-interactive options
APT_OPTIONS="-y -o Dpkg::Options::=--force-confnew -o Dpkg::Options::=--force-confdef"

# Retry logic
apt-get $APT_OPTIONS upgrade || (sleep 10 && apt-get $APT_OPTIONS upgrade)

apt-get $APT_OPTIONS install ufw hostapd dnsmasq unattended-upgrades dhcpcd5 libglib2.0-0 ffmpeg v4l-utils policykit-1 curl ca-certificates

# Restrict dhcpcd to wlan0 entirely to avoid race-conditions with NetworkManager on eth0
echo "allowinterfaces wlan0" >> /etc/dhcpcd.conf

# ----------------------------------------------------------------
# WiFi Radio: Always Unblocked (Appliance Mode)
# ----------------------------------------------------------------
# Kernel: rfkill default = unblocked
if ! grep -q "rfkill.default_state=1" /boot/firmware/cmdline.txt; then
    sed -i 's/$/ rfkill.default_state=1/' /boot/firmware/cmdline.txt
fi

# Disable rfkill state persistence (intentional for appliance)
# Use symlinks instead of systemctl mask (chroot-safe)
ln -sf /dev/null /etc/systemd/system/systemd-rfkill.service
ln -sf /dev/null /etc/systemd/system/systemd-rfkill.socket

# NM: completely ignore wlan0
mkdir -p /etc/NetworkManager/conf.d
cat > /etc/NetworkManager/conf.d/99-unmanaged-wlan.conf << 'EOF'
[device]
match-device=interface-name:wlan0
managed=0
EOF

# 2. User Hygiene
# ----------------------------------------------------------------------
echo "Configuring Users..."
killall -u pi || true
deluser --remove-home pi || true

if ! id "watchmybirds" &>/dev/null; then
    useradd -r -m -d /opt/app/data -s /usr/sbin/nologin -G video,gpio,plugdev watchmybirds
fi
# Sudoers entry handled by Polkit rules below (NoNewPrivileges=true blocks sudo anyway)
# echo "watchmybirds ALL=(ALL) NOPASSWD: /usr/sbin/shutdown, /usr/sbin/reboot" > /etc/sudoers.d/020_watchmybirds-power
# chmod 440 /etc/sudoers.d/020_watchmybirds-power

# Allow watchmybirds to poweroff/reboot via logind (no sudo, headless-safe)
mkdir -p /etc/polkit-1/rules.d
cat > /etc/polkit-1/rules.d/10-watchmybirds-power.rules << 'EOF'
polkit.addRule(function(action, subject) {
    if (subject.user == "watchmybirds" &&
        (action.id == "org.freedesktop.login1.power-off" ||
         action.id == "org.freedesktop.login1.power-off-multiple-sessions" ||
         action.id == "org.freedesktop.login1.reboot" ||
         action.id == "org.freedesktop.login1.reboot-multiple-sessions")) {
        return polkit.Result.YES;
    }
});
EOF
chmod 644 /etc/polkit-1/rules.d/10-watchmybirds-power.rules

# Create admin user (locked by default, for SSH access)
if ! id "admin" &>/dev/null; then
    useradd -m -s /bin/bash -G sudo,video,plugdev admin
    passwd -l admin
fi

passwd -l root

# 2b. Install go2rtc Runtime (native service for relay mode)
# ----------------------------------------------------------------------
echo "Installing go2rtc runtime..."
GO2RTC_VERSION="${GO2RTC_VERSION:-1.9.14}"
GO2RTC_URL="https://github.com/AlexxIT/go2rtc/releases/download/v${GO2RTC_VERSION}/go2rtc_linux_arm64"

curl -fsSL "${GO2RTC_URL}" -o /usr/local/bin/go2rtc
chmod 755 /usr/local/bin/go2rtc

# Shared config path used by both app + go2rtc service.
install -d -o watchmybirds -g watchmybirds -m 0750 /opt/app/data/output
if [ ! -f /opt/app/data/output/go2rtc.yaml ]; then
cat > /opt/app/data/output/go2rtc.yaml << 'EOF'
streams:
  camera: []
api:
  listen: ":1984"
rtsp:
  listen: ":8554"
webrtc:
  listen: ":8555"
EOF
fi
chown watchmybirds:watchmybirds /opt/app/data/output/go2rtc.yaml
chmod 640 /opt/app/data/output/go2rtc.yaml

# 3. Network Security (UFW Prep)
# ----------------------------------------------------------------------
echo "Configuring UFW Defaults..."
sed -i 's/DEFAULT_INPUT_POLICY="ACCEPT"/DEFAULT_INPUT_POLICY="DROP"/' /etc/default/ufw
sed -i 's/IPV6=yes/IPV6=no/' /etc/default/ufw

# 4. SSH Configuration
# ----------------------------------------------------------------------
echo "Configuring SSH..."
sed -i 's/#PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/UsePAM yes/UsePAM yes/' /etc/ssh/sshd_config

# Remove Host Keys
rm -f /etc/ssh/ssh_host_*

# 5. Services (Offline Enablement)
# ----------------------------------------------------------------------
echo "Managing Services (Offline)..."

# DISABLE SSH
rm -f /etc/systemd/system/multi-user.target.wants/ssh.service
rm -f /etc/systemd/system/sshd.service

# DISABLE First-Boot Wizard & Interactive Setup
rm -f /etc/systemd/system/multi-user.target.wants/userconf.service
rm -f /etc/systemd/system/multi-user.target.wants/piwiz.service
systemctl mask userconf.service
systemctl mask piwiz.service
systemctl mask console-setup.service
systemctl mask keyboard-setup.service
systemctl mask systemd-firstboot.service

# INSTALL & ENABLE first-boot hardening
cp /tmp/systemd/wmb-first-boot.service /etc/systemd/system/wmb-first-boot.service
chmod 644 /etc/systemd/system/wmb-first-boot.service
ln -s /etc/systemd/system/wmb-first-boot.service /etc/systemd/system/multi-user.target.wants/wmb-first-boot.service

# Copy First-Boot Script
cp /tmp/first-boot/first-boot.sh /usr/local/bin/wmb-first-boot.sh
chmod 755 /usr/local/bin/wmb-first-boot.sh

# Install WiFi setup path/service and helper
cp /tmp/systemd/wmb-wifi-setup.service /etc/systemd/system/wmb-wifi-setup.service
cp /tmp/systemd/wmb-wifi-setup.path /etc/systemd/system/wmb-wifi-setup.path
chmod 644 /etc/systemd/system/wmb-wifi-setup.service
chmod 644 /etc/systemd/system/wmb-wifi-setup.path
cp /tmp/first-boot/setup_wifi.sh /usr/local/bin/wmb-setup-wifi.sh
chmod 755 /usr/local/bin/wmb-setup-wifi.sh
if [ ! -e /etc/systemd/system/multi-user.target.wants/wmb-wifi-setup.path ]; then
    ln -s /etc/systemd/system/wmb-wifi-setup.path /etc/systemd/system/multi-user.target.wants/wmb-wifi-setup.path
fi

# Install boot diagnostics service (writes logs to /boot/firmware/debuglogs)
cp /tmp/systemd/wmb-boot-diag.service /etc/systemd/system/wmb-boot-diag.service
cp /tmp/systemd/wmb-boot-diag.sh /usr/local/bin/wmb-boot-diag.sh
chmod 644 /etc/systemd/system/wmb-boot-diag.service
chmod 755 /usr/local/bin/wmb-boot-diag.sh
ln -sf /etc/systemd/system/wmb-boot-diag.service /etc/systemd/system/multi-user.target.wants/wmb-boot-diag.service

# Install WiFi watchdog (Auto-Fallback to AP)
cp /tmp/systemd/wmb-wifi-watchdog.sh /usr/local/bin/wmb-wifi-watchdog.sh
chmod 755 /usr/local/bin/wmb-wifi-watchdog.sh
cp /tmp/systemd/wmb-wifi-watchdog.service /etc/systemd/system/wmb-wifi-watchdog.service
cp /tmp/systemd/wmb-wifi-watchdog.timer /etc/systemd/system/wmb-wifi-watchdog.timer
chmod 644 /etc/systemd/system/wmb-wifi-watchdog.service
chmod 644 /etc/systemd/system/wmb-wifi-watchdog.timer
# Enable the timer, NOT the service (timer triggers service)
ln -sf /etc/systemd/system/wmb-wifi-watchdog.timer /etc/systemd/system/multi-user.target.wants/wmb-wifi-watchdog.timer

# Install setup server (AP only; enabled by first-boot when needed)
cp /tmp/systemd/wmb-setup-server.service /etc/systemd/system/wmb-setup-server.service
chmod 644 /etc/systemd/system/wmb-setup-server.service
cp /tmp/setup-server/setup_server.py /usr/local/bin/wmb-setup-server.py
chmod 755 /usr/local/bin/wmb-setup-server.py
mkdir -p /usr/local/share/watchmybirds/setup/templates
cp /tmp/setup-server/templates/setup.html /usr/local/share/watchmybirds/setup/templates/setup.html

# Install go2rtc systemd service
cat > /etc/systemd/system/go2rtc.service << 'EOF'
[Unit]
Description=go2rtc media relay
After=network-online.target
Wants=network-online.target

[Service]
User=watchmybirds
Group=watchmybirds
WorkingDirectory=/opt/app
ExecStart=/usr/local/bin/go2rtc -config /opt/app/data/output/go2rtc.yaml
Restart=always
RestartSec=2
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/app/data/output
ProtectHome=yes

[Install]
WantedBy=multi-user.target
EOF
chmod 644 /etc/systemd/system/go2rtc.service
ln -sf /etc/systemd/system/go2rtc.service /etc/systemd/system/multi-user.target.wants/go2rtc.service

# DISABLE AP Services
rm -f /etc/systemd/system/multi-user.target.wants/hostapd.service
rm -f /etc/systemd/system/multi-user.target.wants/dnsmasq.service

# 6. Configuration Templates
# ----------------------------------------------------------------------
echo "Installing Templates..."
cp /tmp/ap/hostapd.conf.template /etc/hostapd/hostapd.conf
cp /tmp/ap/dnsmasq.conf.template /etc/dnsmasq.conf

# 7. Cleanup
# ----------------------------------------------------------------------
echo "Cleaning up..."
apt-get clean
rm -rf /var/lib/apt/lists/*
cat /dev/null > /root/.bash_history

echo "Hardening Complete."
