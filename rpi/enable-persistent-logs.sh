#!/usr/bin/env bash

# ------------------------------------------------------------------------------
# Logging Upgrade: Enable Persistent Journald
# ------------------------------------------------------------------------------
# This script configures systemd-journald to store logs permanently on disk.
# It sets a size limit to prevent the SD card from filling up.
#
# RUN THIS ON THE RPI:
# bash rpi/enable-persistent-logs.sh
# ------------------------------------------------------------------------------

set -euo pipefail

echo "--> Configuring persistent journal logging..."

# 1. Create directory if it doesn't exist (triggers systemd to use it)
sudo mkdir -p /var/log/journal

# 2. Fix permissions (systemd-journal group needs access)
sudo systemd-tmpfiles --create --prefix /var/log/journal

# 3. Configure journald persistence using a drop-in.
sudo mkdir -p /etc/systemd/journald.conf.d
cat <<'EOF' | sudo tee /etc/systemd/journald.conf.d/persistence.conf >/dev/null
[Journal]
Storage=persistent
SystemMaxUse=500M
EOF

echo "--> Configuration updated."

# 4. Restart journald to apply changes
echo "--> Restarting systemd-journald..."
sudo systemctl restart systemd-journald
sudo journalctl --flush || true

# 5. Verification
if [ -d "/var/log/journal" ]; then
    echo "✅ Success! Persistent logging enabled."
    echo "   Logs will persist across reboots in /var/log/journal/"
    echo "   View them with: journalctl --boot=-1 (for previous boot)"
else
    echo "❌ Error: /var/log/journal/ not created."
    exit 1
fi
