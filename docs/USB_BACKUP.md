# USB Backup (write-only v1)

WatchMyBirds writes daily snapshots of the SQLite database, captured frames,
and the installed app code to a USB stick mounted at `/mnt/wmb-backup`. This
is your protection against SD-card death, which is the single most common
hardware failure on a long-running Raspberry Pi.

> **v1 is write-only.** Restore is currently a manual procedure (mount the
> stick on any Linux machine and copy files back). A UI-driven restore flow
> ships in a follow-up release together with the OTA pre-update snapshot
> hook. See *Recovery* below.

## What you need

- Any USB stick **at least 2× the size of your SD card** (rule of thumb:
  daily snapshots use hardlinks for unchanged frames, so total stick usage
  is roughly *live data + 1 full copy + a few % drift per day*).
- A Linux/macOS machine to format the stick. (Windows can't write ext4
  natively; use a Live USB or WSL.)

## One-time setup

> **Warning:** the format step **destroys all data on the stick**. Make
> sure you've picked the right device.

1. Plug the stick into your Mac/Linux box (not the Pi yet) and find its
   device node:

   ```bash
   lsblk           # Linux
   diskutil list   # macOS
   ```

   Look for the new device, e.g. `/dev/sdb`. **Use the whole-disk node**
   (`/dev/sdb`), not a partition (`/dev/sdb1`), unless you already have a
   partition layout you want to keep.

2. Format with ext4 and the label `WMB-BACKUP`:

   ```bash
   # Linux
   sudo mkfs.ext4 -L WMB-BACKUP /dev/sdb
   ```

   The label is **mandatory** — the Pi mounts the stick by label, not by
   device path, so it survives re-plugging into a different USB port.

3. Eject cleanly, plug into the Pi.

That's it. The systemd automount unit picks it up on first access, the
daily timer runs at 03:00, and the Settings UI surfaces the stick state.

## Why ext4?

The backup script uses `rsync --link-dest` so daily snapshots share
hardlinks for unchanged frames. **Without hardlinks, every snapshot would
be a full copy** — the stick fills up in days instead of months. FAT32 and
exFAT don't support hardlinks. NTFS does, but its Linux driver is too slow
and fragile for unattended overnight runs.

The mount is also locked down: `nosuid,nodev,noexec` means even if someone
plants an executable on the stick, the Pi refuses to run it.

## Recovery (manual, v1)

Until the UI-driven restore lands, copy data back manually:

1. Power off the Pi (or unmount cleanly: `sudo umount /mnt/wmb-backup`).
2. Pull the stick. Plug into a Linux machine.
3. Mount it: `sudo mount /dev/disk/by-label/WMB-BACKUP /mnt`
4. Browse `/mnt/snapshots/` — directories are named
   `YYYYMMDD_HHMMSS_<kind>/` where `<kind>` is `scheduled` or `manual`.
5. Each snapshot directory contains:
   - `data/images.db` — SQLite database (already verified with
     `pragma integrity_check`; see `manifest.json`)
   - `data/frames/` — captured frames
   - `app/` — the installed app code, useful only if you also want to
     pin to that exact build
   - `manifest.json` — what was captured, sizes, integrity hashes
   - `COMPLETED` — marker file. **Trust no snapshot directory that lacks
     this file** — it crashed mid-write.

The newest valid snapshot is also reachable via the symlink
`/mnt/latest/`.

To restore onto a fresh Pi installation:

```bash
sudo systemctl stop app.service
sudo cp /mnt/<snapshot>/data/images.db /opt/app/data/images.db
sudo rsync -a /mnt/<snapshot>/data/frames/ /opt/app/data/frames/
sudo chown -R watchmybirds:watchmybirds /opt/app/data
sudo systemctl start app.service
```

Do **not** restore the `app/` tree onto a running install unless you
really want to roll back the app to the snapshotted version — for that
case, use OTA's rollback flow (when it ships) instead.

## What is NOT backed up (v1)

- Audio recordings (audio is currently archived as a feature; will be
  added separately if/when audio returns to mainline)
- `/opt/app/.venv/` — pip-rebuilt on every release, no point copying
- System config (`/etc/`, network settings, SSH keys, wifi credentials)
  — these are baked into the image, not the backup
- Encrypted secrets at rest — v1 stick is plain ext4. If your threat
  model includes "someone steals the stick", encrypt at the volume level
  yourself (LUKS) before formatting; the mount unit accepts any ext4
  volume regardless of underlying encryption.

## Status & troubleshooting

The Settings page surfaces:

- **Stick connected / missing** — automount only mounts `WMB-BACKUP`-labelled
  ext4 volumes; anything else is reported as `wrong-fs` with format
  instructions.
- **Free space + warning at >80% full** — old snapshots beyond the
  retention window are pruned automatically (7 daily, 4 weekly, 6 monthly
  for scheduled; latest 3 for manual triggers).
- **Last 5 snapshots** — with completion state and verification hash.

If the Pi reports `wrong-fs` repeatedly, the most common cause is a stick
that came pre-formatted as exFAT or FAT32 from the factory. Reformat as
above.

If the stick keeps showing as missing after replugging, run on the Pi:

```bash
journalctl -u 'mnt-wmb*backup*' --since '10 min ago'
```

This is also the right thing to attach when reporting a backup-related
issue.
