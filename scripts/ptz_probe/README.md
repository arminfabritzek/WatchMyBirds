# PTZ Capability Probe

Operator-attended CLI tool that finds out what an ONVIF PTZ camera
**actually** can do — not what marketing claims, not what cheap-cam apps
imply, but what its ONVIF service really exposes and what it really does
under each move command.

Output:

- A structured YAML report + JSON twin + raw log per run (next to the
  script)
- A canonical empirical-capabilities cache file at
  `OUTPUT_DIR/ptz_capabilities/cam<id>.yaml` — this is what WMB's
  Settings UI reads to render its tri-state capability pills

## Quick start

From the WMB repo root, with the WMB venv active:

```bash
python -m scripts.ptz_probe \
    --ip 192.168.1.100 --user admin \
    --camera-id 0
```

You'll be prompted for:

- The password (or set `WMB_CAM_PASSWORD` in env)
- A Home preset (the probe needs a safe return point between tests)
- `y` / `n` / `s` (skip) for each move test as you watch the camera

When the probe finishes, the cam returns to Home, the reports get
written, and — because `--camera-id` is set — WMB's Settings UI will
pick up the empirical results on the next page refresh.

## Where to run it

### On your Mac (developing or troubleshooting from your desk)

```bash
cd <wmb-repo-root>
source .venv/bin/activate
python -m scripts.ptz_probe --ip 192.168.1.100 --user admin --camera-id 0
```

OUTPUT_DIR is auto-resolved from WMB's `config.get_config()` — usually
`./data/output` in a dev checkout.

### On the Raspberry Pi (the deployed appliance)

```bash
ssh admin@<rpi-host>
cd /opt/app
sudo -u watchmybirds python -m scripts.ptz_probe \
    --ip 192.168.1.100 --user admin --camera-id 0
```

`/opt/app/data/` is owned by the `watchmybirds` user (0750), so the
probe needs to be run with that user's permissions to write to
`/opt/app/data/ptz_capabilities/`. Sudo + `-u watchmybirds` is the
clean way; alternatively, run it from your Mac and `scp` the resulting
`cam0.yaml` to the RPi.

### As a non-WMB checkout (rare — only when debugging the tool itself)

The probe also works without WMB on `sys.path`. Then `--output-dir`
becomes mandatory if you want the canonical cache file written:

```bash
python scripts/ptz_probe/__main__.py \
    --ip 192.168.1.100 --user admin \
    --camera-id 0 \
    --output-dir /path/to/wmb/data/output
```

## What the probe answers per camera

- What does it **claim** via `GetServiceCapabilities` + `GetNodes`?
  (This is the declared side — also visible in WMB Settings as the
  baseline pill colour.)
- What does it actually **do** when issued continuous / relative /
  absolute moves with small and large amplitudes? (This is the empirical
  side — green/red pills come from here.)
- Does `GetStatus` return a real position? Does `MoveStatus` actually
  transition `MOVING → IDLE` or is it stuck on one value?
- Is `RelativeMove` reproducible (same input → same delta)?

## Safety

This tool issues **real PTZ commands to real hardware**. Mitigations:

- **Mandatory Home preset.** The probe will not run move tests unless a
  Home preset exists. It asks you to set one first if missing. Between
  every test block it returns to Home automatically. On exit it always
  returns to Home one last time.
- **Ctrl-C is wired to emergency stop.** Pressing it issues an immediate
  ONVIF `Stop(PanTilt=True, Zoom=True)` before exiting.
- **AbsoluteMove is opt-in per run.** Even when declared by the cam, the
  probe asks before issuing absolute moves — some cheap cams interpret
  `AbsoluteMove(0,0)` as "go to the last endstop" rather than centre.
- **Continuous bursts are short by default.** 300 ms with auto-stop.
  One long-burst case (1.5 s) exists to spot timing issues.

## What WMB does with the result

After the probe finishes, the file
`OUTPUT_DIR/ptz_capabilities/cam<id>.yaml` looks like:

```yaml
camera_id: 0
probed_at: 20260517_235927
connection:
  ip: 192.168.1.100
empirical:
  continuous_works: true
  relative_works:   false   # cam declares relative=true via GetNodes,
  absolute_works:   false   # but the probe found it doesn't really work
  movestatus_transitions: false
recommended_strategy: continuous_pulse
```

WMB's Settings UI reads this on every PTZ-modal open and renders pills:

| Pill | Colour | Meaning |
|---|---|---|
| `continuous PT` | 🟢 green ✓ | declared and empirically works |
| `relative PT` | 🔴 red ⚠ | declared by camera, but probe shows it doesn't |
| `home` | ⬜ gray ✗ | not declared |
| `MoveStatus` | 🟡 yellow ? | declared, never empirically probed yet |

Until you run the probe, every empirically-testable pill is yellow.
After a successful probe run, every pill becomes green or red.

## Flags

| Flag | Purpose |
|---|---|
| `--ip` / `--port` / `--user` / `--password` | Cam connection. `--password` omitted = interactive `getpass`. Or set `WMB_CAM_PASSWORD` env. |
| `--camera-id` | WMB camera id. Required to write the canonical cache. |
| `--output-dir` | WMB OUTPUT_DIR. Optional — auto-resolved via WMB's `config.get_config()` if not set. |
| `--verbose` / `-v` | Also stream debug logs (including zeep traffic) to stderr. |
| `--no-prompts` | Skip Y/N prompts where a sensible default exists. Move tests still ask for feedback per case. |
| `--skip-continuous` / `--skip-relative` / `--skip-absolute` | Skip a specific move-test block. |

## Files

| File | Purpose |
|---|---|
| `__main__.py` | CLI front-end + test sequencer + report writer |
| `core.py` | Pure ONVIF operations, dataclasses, zero UI |
| `README.md` | This file |
| `__init__.py` | Marks the directory as a Python package |
| `.ptz_probe_settings.json` | Last-used IP/port/user (no password) — gitignored |
| `ptz_probe_report_*.yaml/.json/.log` | Per-run reports — gitignored |

## See also

- WMB Settings UI: capability pills consume the canonical cache file
- `core/ptz_core.py::probe_capabilities` + `_load_empirical_from_disk`:
  the read side of this contract
- `docs/INVARIANTS.md`: PTZ-related hard invariants
- In-UI probe wizard: writes to the same canonical path
