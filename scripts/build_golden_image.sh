#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <repo-root> [output-image]" >&2
  exit 1
fi

repo_root="$(cd "$1" && pwd)"
output_img="${2:-golden.img}"
mount_point="$(pwd)/mount_point"
source_archive="$(pwd)/rpios.img.xz"
base_url="https://downloads.raspberrypi.com/raspios_lite_arm64/images/raspios_lite_arm64-2025-12-04/2025-12-04-raspios-trixie-arm64-lite.img.xz"
loop_dev=""

cleanup_mount() {
  local target="$1"
  local attempt

  if ! mountpoint -q "$target"; then
    return 0
  fi

  for attempt in 1 2 3; do
    if sudo umount "$target"; then
      return 0
    fi
    echo "umount failed for $target (attempt ${attempt}/3); killing remaining processes..."
    sudo fuser -km "$target" 2>/dev/null || true
    sleep 2
  done

  echo "umount still failed for $target; falling back to lazy unmount"
  sudo umount -l "$target"
}

cleanup() {
  local exit_code=$?

  set +e
  sync

  if [[ -d "$mount_point" ]]; then
    sudo fuser -km "$mount_point" 2>/dev/null || true
    sleep 1

    sudo rm -f "$mount_point/usr/bin/qemu-aarch64-static" 2>/dev/null || true
    sudo rm -f "$mount_point/tmp/harden.sh" 2>/dev/null || true
    sudo rm -f "$mount_point/tmp/install-python312.sh" 2>/dev/null || true
    sudo rm -rf "$mount_point/tmp/first-boot" 2>/dev/null || true
    sudo rm -rf "$mount_point/tmp/systemd" 2>/dev/null || true
    sudo rm -rf "$mount_point/tmp/polkit" 2>/dev/null || true
    sudo rm -rf "$mount_point/tmp/ap" 2>/dev/null || true
    sudo rm -rf "$mount_point/tmp/setup-server" 2>/dev/null || true

    cleanup_mount "$mount_point/etc/resolv.conf"
    cleanup_mount "$mount_point/proc"
    cleanup_mount "$mount_point/sys"
    cleanup_mount "$mount_point/dev/pts"
    cleanup_mount "$mount_point/dev"
    cleanup_mount "$mount_point/boot/firmware"
    cleanup_mount "$mount_point"
  fi

  if [[ -n "$loop_dev" ]]; then
    sudo losetup -d "$loop_dev" 2>/dev/null || true
  fi

  exit "$exit_code"
}

trap cleanup EXIT

rm -f "$source_archive" "$output_img"
rm -rf "$mount_point"

echo "Downloading Raspberry Pi OS Lite base image..."
wget -O "$source_archive" "$base_url"
unxz "$source_archive"
mv ./*.img "$output_img"

echo "Expanding image by 4 GiB..."
dd if=/dev/zero bs=1M count=4096 >> "$output_img"

echo "Resizing partition table..."
sudo parted -s "$output_img" resizepart 2 100%

echo "Growing filesystem..."
loop_dev="$(sudo losetup --find --show --partscan "$output_img")"
sudo e2fsck -f -p "${loop_dev}p2"
sudo resize2fs "${loop_dev}p2"
sudo losetup -d "$loop_dev"
loop_dev=""

echo "Applying first-boot assets and production hardening..."
mkdir -p "$mount_point"
loop_dev="$(sudo losetup --find --show --partscan "$output_img")"

sudo mount "${loop_dev}p2" "$mount_point"
sudo mount "${loop_dev}p1" "$mount_point/boot/firmware"

sudo cp /usr/bin/qemu-aarch64-static "$mount_point/usr/bin/"
sudo cp "$repo_root/rpi/harden.sh" "$mount_point/tmp/"
sudo mkdir -p \
  "$mount_point/tmp/first-boot" \
  "$mount_point/tmp/systemd" \
  "$mount_point/tmp/polkit" \
  "$mount_point/tmp/ap" \
  "$mount_point/tmp/setup-server/templates"
sudo cp "$repo_root/rpi/first-boot/first-boot.sh" "$mount_point/tmp/first-boot/"
sudo cp "$repo_root/rpi/first-boot/setup_wifi.sh" "$mount_point/tmp/first-boot/"
sudo cp "$repo_root/rpi/systemd/wmb-first-boot.service" "$mount_point/tmp/systemd/"
sudo cp "$repo_root/rpi/systemd/wmb-wifi-setup.service" "$mount_point/tmp/systemd/"
sudo cp "$repo_root/rpi/systemd/wmb-wifi-setup.path" "$mount_point/tmp/systemd/"
sudo cp "$repo_root/rpi/systemd/wmb-setup-server.service" "$mount_point/tmp/systemd/"
sudo cp "$repo_root/rpi/systemd/wmb-wifi-watchdog.sh" "$mount_point/tmp/systemd/"
sudo cp "$repo_root/rpi/systemd/wmb-wifi-watchdog.service" "$mount_point/tmp/systemd/"
sudo cp "$repo_root/rpi/systemd/wmb-wifi-watchdog.timer" "$mount_point/tmp/systemd/"
sudo cp "$repo_root/rpi/systemd/wmb-boot-diag.service" "$mount_point/tmp/systemd/"
sudo cp "$repo_root/rpi/systemd/wmb-boot-diag.sh" "$mount_point/tmp/systemd/"
# USB backup volume (Phase 1 of usb-data-backup plan).
# The literal '\x2d' in the filename is the systemd-escape for '-' inside
# a path segment ('-' inside the unit name is reserved as a path separator).
# Produced by: systemd-escape -p /mnt/wmb-backup
sudo cp "$repo_root/rpi/systemd/mnt-wmb\\x2dbackup.mount" "$mount_point/tmp/systemd/"
sudo cp "$repo_root/rpi/systemd/mnt-wmb\\x2dbackup.automount" "$mount_point/tmp/systemd/"
# Scheduled backup service + timer (Phase 3 of usb-data-backup plan).
sudo cp "$repo_root/rpi/systemd/wmb-backup.service" "$mount_point/tmp/systemd/"
sudo cp "$repo_root/rpi/systemd/wmb-backup.timer" "$mount_point/tmp/systemd/"
# One-shot USB stick formatter (root, polkit-gated to watchmybirds).
sudo cp "$repo_root/rpi/systemd/wmb-format-backup.service" "$mount_point/tmp/systemd/"
sudo cp "$repo_root/rpi/polkit/10-watchmybirds-format-backup.rules" "$mount_point/tmp/polkit/"
# OTA updater (oneshot, root, polkit-gated to watchmybirds for start).
sudo cp "$repo_root/rpi/systemd/wmb-update.service" "$mount_point/tmp/systemd/"
sudo cp "$repo_root/rpi/polkit/10-watchmybirds-update.rules" "$mount_point/tmp/polkit/"
sudo cp "$repo_root/rpi/ap/hostapd.conf.template" "$mount_point/tmp/ap/"
sudo cp "$repo_root/rpi/ap/dnsmasq.conf.template" "$mount_point/tmp/ap/"
sudo cp "$repo_root/rpi/install-python312.sh" "$mount_point/tmp/"
sudo cp "$repo_root/rpi/setup-server/setup_server.py" "$mount_point/tmp/setup-server/"
sudo cp "$repo_root/rpi/setup-server/templates/setup.html" "$mount_point/tmp/setup-server/templates/"

sudo chmod +x "$mount_point/tmp/harden.sh"
sudo chmod +x "$mount_point/tmp/install-python312.sh"

sudo mount --bind /dev "$mount_point/dev"
sudo mount --bind /dev/pts "$mount_point/dev/pts"
sudo mount --bind /sys "$mount_point/sys"
sudo mount --bind /proc "$mount_point/proc"
sudo mount --bind /etc/resolv.conf "$mount_point/etc/resolv.conf"

sudo chroot "$mount_point" /bin/bash -c "/tmp/harden.sh"

echo "Golden image ready at $output_img"
