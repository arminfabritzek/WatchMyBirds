# PTZ laboratory

These probes inspect an ONVIF camera without issuing movement, focus, preset,
home-position, stop, or imaging-setting commands. Reports are written as JSON
and YAML below `output/ptz_lab/`; credentials are removed before serialization.

Run from the repository root. The password is prompted without echo:

```bash
.venv/bin/python -m scripts.ptz_lab.test_ptz_identity \
  --ip CAMERA_IP --user YOUR_ONVIF_USER

.venv/bin/python -m scripts.ptz_lab.test_ptz_capabilities \
  --ip CAMERA_IP --user YOUR_ONVIF_USER

.venv/bin/python -m scripts.ptz_lab.test_ptz_status \
  --ip CAMERA_IP --user YOUR_ONVIF_USER --samples 10 --interval 0.25
```

For unattended local use, put the password in `WMB_CAM_PASSWORD` for the life
of that shell. Do not put it in a command argument, report, or tracked file.

The application on the Raspberry Pi may remain active during these three
read-only probes. Stop it before running any future movement probe so WMB and
the laboratory cannot control the camera at the same time.

The first physical probe is deliberately limited to speed `0.25` and 250 ms.
It requires an explicit execution flag, always sends an emergency stop, and
returns to the supplied preset in a `finally` block:

```bash
.venv/bin/python -m scripts.ptz_lab.test_ptz_continuous \
  --ip CAMERA_IP --port ONVIF_PORT --user YOUR_ONVIF_USER \
  --return-preset YOUR_PRESET_TOKEN --axis pan --direction 1 \
  --speed 0.10 --duration-ms 100 --execute
```

Larger operator-attended tests retain the 2-second client cap but require an
additional explicit acknowledgement:

```bash
.venv/bin/python -m scripts.ptz_lab.test_ptz_continuous \
  --ip CAMERA_IP --port ONVIF_PORT --user YOUR_ONVIF_USER \
  --return-preset YOUR_PRESET_TOKEN --axis pan --direction 1 \
  --speed 1.0 --duration-ms 500 --large-jump --execute
```

To measure how quickly a known point in the overview image can be centered,
use the feedback-controlled pan/tilt probe. Coordinates refer to the first
snapshot; the probe measures displacement after every pulse and stops within
the requested pixel tolerance:

```bash
.venv/bin/python -m scripts.ptz_lab.test_ptz_center_target \
  --ip CAMERA_IP --port ONVIF_PORT --user YOUR_ONVIF_USER \
  --return-preset YOUR_PRESET_TOKEN --target-name example_target \
  --target-x 630 --target-y 224 --tolerance-px 15 --execute
```

This physical probe also sends an emergency stop and returns to the supplied
preset even when centering or snapshot capture fails. Its report distinguishes
total setup time from `control_elapsed_s`, the time between the baseline image
and confirmed centering.

The production follow/recovery sequence can be checked without camera movement.
This simulates detections at configurable low input rates such as 0.5–1 FPS,
asserts the calibrated pan and zoom commands, exercises bounded recovery search,
and verifies the final overview return:

```bash
.venv/bin/python scripts/ptz_lab/test_follow_simulation.py --fps 0.5 1.0
```
