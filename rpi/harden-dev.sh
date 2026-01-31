#!/bin/bash
set -e
set -x

# ----------------------------------------------------------------------
# WatchMyBirds DEV Image Hardening Script
# DEVELOPMENT ONLY - SSH enabled, /opt/app writable
# ----------------------------------------------------------------------

export DEBIAN_FRONTEND=noninteractive
export APT_LISTCHANGES_FRONTEND=none

echo "update_initramfs=no" >> /etc/initramfs-tools/update-initramfs.conf || true

# ----------------------------------------------------------------------
# 0. Logging Persistence (CRITICAL for debugging)
# ----------------------------------------------------------------------
echo "Configuring Persistent Logging..."
mkdir -p /var/log/journal
# Use --root to avoid requiring /proc during image build chroot.
systemd-tmpfiles --root=/ --create --prefix /var/log/journal
# Force persistence in journald.conf
sed -i 's/#Storage=auto/Storage=persistent/' /etc/systemd/journald.conf
sed -i 's/#SystemMaxUse=/SystemMaxUse=100M/' /etc/systemd/journald.conf

# 1. Update & Install Dependencies
# ----------------------------------------------------------------------
apt-get update

APT_OPTIONS="-y -o Dpkg::Options::=--force-confnew -o Dpkg::Options::=--force-confdef"
apt-get $APT_OPTIONS upgrade || (sleep 10 && apt-get $APT_OPTIONS upgrade)
apt-get $APT_OPTIONS install ufw hostapd dnsmasq unattended-upgrades dhcpcd5 libglib2.0-0 ffmpeg v4l-utils openssh-server policykit-1

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

# 2. User Hygiene (DEV: admin unlocked with SSH key)
# ----------------------------------------------------------------------
echo "Configuring Users (DEV MODE)..."
killall -u pi || true
deluser --remove-home pi || true

if ! id "watchmybirds" &>/dev/null; then
    useradd -r -m -d /var/lib/watchmybirds -s /usr/sbin/nologin -G video,gpio,plugdev watchmybirds
fi
# Sudoers entry handled by Polkit rules below
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

# Create admin user (UNLOCKED for dev)
if ! id "admin" &>/dev/null; then
    useradd -m -s /bin/bash -G sudo,video,plugdev admin
fi
# DO NOT lock admin - keep unlocked for SSH access

# Setup SSH Key for admin
mkdir -p /home/admin/.ssh
if [ -f /tmp/dev_authorized_keys ]; then
    cp /tmp/dev_authorized_keys /home/admin/.ssh/authorized_keys
    chown -R admin:admin /home/admin/.ssh
    chmod 700 /home/admin/.ssh
    chmod 600 /home/admin/.ssh/authorized_keys
    echo "SSH Key installed for admin user"
else
    echo "WARNING: No dev_authorized_keys found - SSH key auth will not work!"
fi

# GRANT ADMIN ROOT PRIVILEGES (NOPASSWD) FOR DEV CONVENIENCE
echo "admin ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/010_admin-nopasswd
chmod 440 /etc/sudoers.d/010_admin-nopasswd

passwd -l root

# 3. Network Security (UFW Prep - SSH allowed)
# ----------------------------------------------------------------------
echo "Configuring UFW Defaults (DEV: SSH allowed)..."
sed -i 's/DEFAULT_INPUT_POLICY="ACCEPT"/DEFAULT_INPUT_POLICY="DROP"/' /etc/default/ufw
sed -i 's/IPV6=yes/IPV6=no/' /etc/default/ufw

# 4. SSH Configuration (ENABLED for dev)
# ----------------------------------------------------------------------
echo "Configuring SSH (DEV: ENABLED)..."
sed -i 's/#PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
# Password auth disabled - key only
sed -i 's/#PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/UsePAM yes/UsePAM yes/' /etc/ssh/sshd_config

rm -f /etc/ssh/ssh_host_*

# 5. Services (DEV: SSH ENABLED)
# ----------------------------------------------------------------------
echo "Managing Services (DEV MODE)..."

# ENABLE SSH (different from production!)
ln -sf /lib/systemd/system/ssh.service /etc/systemd/system/multi-user.target.wants/ssh.service

# Disable First-Boot Wizard & Interactive Setup
rm -f /etc/systemd/system/multi-user.target.wants/userconf.service
rm -f /etc/systemd/system/multi-user.target.wants/piwiz.service
systemctl mask userconf.service
systemctl mask piwiz.service
systemctl mask console-setup.service
systemctl mask keyboard-setup.service
systemctl mask systemd-firstboot.service

# Install first-boot hardening
cp /tmp/systemd/wmb-first-boot.service /etc/systemd/system/wmb-first-boot.service
chmod 644 /etc/systemd/system/wmb-first-boot.service
ln -sf /etc/systemd/system/wmb-first-boot.service /etc/systemd/system/multi-user.target.wants/wmb-first-boot.service

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
    ln -sf /etc/systemd/system/wmb-wifi-setup.path /etc/systemd/system/multi-user.target.wants/wmb-wifi-setup.path
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
# Enable the timer
ln -sf /etc/systemd/system/wmb-wifi-watchdog.timer /etc/systemd/system/multi-user.target.wants/wmb-wifi-watchdog.timer

# Install setup server
cp /tmp/systemd/wmb-setup-server.service /etc/systemd/system/wmb-setup-server.service
chmod 644 /etc/systemd/system/wmb-setup-server.service
cp /tmp/setup-server/setup_server.py /usr/local/bin/wmb-setup-server.py
chmod 755 /usr/local/bin/wmb-setup-server.py
mkdir -p /usr/local/share/watchmybirds/setup/templates
cp /tmp/setup-server/templates/setup.html /usr/local/share/watchmybirds/setup/templates/setup.html

# Disable AP Services
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

echo "=============================================="
echo "DEV MODE Hardening Complete."
echo "SSH: ENABLED (Key Auth Only)"
echo "Admin User: UNLOCKED"
echo "=============================================="
