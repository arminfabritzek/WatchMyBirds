# Visual PTZ MVP

This experimental MVP returns a cheap ONVIF PTZ camera to a known view by
looking at the scene instead of relying on a user-created camera preset.
It is deliberately CPU-first and runs on an Apple M1 as well as the normal
WatchMyBirds targets.

## What it proves

1. Capture the current view as a visual keyframe.
2. Detect SIFT features (ORB is available as a fallback).
3. Match the current image to the keyframe.
4. Reject outliers with RANSAC and estimate a homography.
5. Convert the remaining image-space error into a bounded PTZ burst.
6. Capture another frame and verify that the visual error improved.
7. Stop on alignment, poor match quality, divergence, iteration limit, or
   cumulative motion budget.

Keyframes and debug images live under
`OUTPUT_DIR/derivatives/visual_ptz/cam<ID>/`. They are regenerable derivative
assets; bird originals and the detections database are not touched.

## Offline check on a Mac

From the repository virtualenv:

```bash
python -m scripts.visual_ptz_mvp demo \
  --output-dir output/visual_ptz_demo
```

The command creates `reference.jpg`, a displaced `current.jpg`, and
`matches.jpg`. The last image draws only RANSAC inliers in green and prints
the measured error, quality, and inlier count. The JSON output should report
approximately `pixel_dx=85` and `pixel_dy=-45`.

## Test with a cheap ONVIF PTZ camera

Pre-flight:

- Add the camera in WatchMyBirds Settings and verify the live stream.
- Verify manual pan/tilt controls first.
- Turn **Auto-PTZ off** so the existing tracker cannot fight the experiment.
- Start near the middle of the mechanical range, away from an end stop.
- Keep a hand on power for the first real movement test.

### Click-to-Aim in the live view

The normal appliance UI is the easiest end-to-end test and does not require
an ONVIF preset:

1. Run WatchMyBirds on the Raspberry Pi as usual and open the **Live** page.
2. Hover the stream and click **◎ Aim** in the PTZ control.
3. Click a textured point in the video, such as the edge of a feeder or branch.
4. Watch the target ring and the status sequence: **Matching → Moving →
   Verifying → Centered**.
5. Press **Esc**, click **Aim** again, or touch a manual PTZ button to stop.

The Raspberry Pi reads its own current camera frames and performs the OpenCV
matching locally. A Mac is not required at runtime. During this short-lived
session, auto-PTZ is paused exclusively and resumed afterward even when the
aim fails. The first version uses pan and tilt only, caps the run at five
corrections, and refuses further motion below the visual quality threshold.

The status chip exposes the measured quality percentage, RANSAC inlier count,
and current iteration. **Target lost** with a low score means the safety gate
worked; choose a point with more stable texture and overlap. No frame or click
coordinate is written to the database.

For the first hardware run, keep auto-PTZ switched off anyway so the new
behavior can be observed in isolation. Once direction and response are
verified, the Aim controller can safely exercise its automatic pause/resume
path with auto-PTZ enabled.

Point the camera at an interesting view with the existing joystick. This is
only how the MVP obtains its first visual anchor; no ONVIF preset is created.

```bash
python -m scripts.visual_ptz_mvp capture \
  --camera-id 0 \
  --keyframe feeder
```

Move the camera away with the joystick. First perform a read-only match:

```bash
python -m scripts.visual_ptz_mvp align \
  --camera-id 0 \
  --keyframe feeder
```

Inspect the printed quality, error, and `dry_run.jpg`. If pan or tilt signs are
opposite for this camera, add `--invert-pan` or `--invert-tilt`.

To identify the current place across all captured anchors instead of naming a
keyframe up front, run:

```bash
python -m scripts.visual_ptz_mvp recognize --camera-id 0
```

Then allow real, bounded movements toward a selected keyframe:

```bash
python -m scripts.visual_ptz_mvp align \
  --camera-id 0 \
  --keyframe feeder \
  --execute \
  --max-iterations 6 \
  --max-step 0.15 \
  --max-total-motion 0.8
```

Every iteration writes a match image and `report.json`. The report's
`progress` field is the reduction in normalized visual error since the
previous frame. Two non-improving corrections stop the run automatically.
`Ctrl-C` requests an emergency stop.

## Verify commanded motion from the live video

`probe-motion` captures a frame, sends one bounded command, captures the next
frame, and measures what the camera physically did. Pan and tilt are reported
as a fraction of the frame; zoom is reported as a scale change. The command
also recommends a burst count needed to reach the requested response.

Start with a no-move baseline:

```bash
python -m scripts.visual_ptz_mvp probe-motion \
  --camera-id 0 --axis pan --direction 1
```

Then allow three conservative real pulses:

```bash
python -m scripts.visual_ptz_mvp probe-motion \
  --camera-id 0 --axis pan --direction 1 \
  --pulses 3 --speed 0.25 --duration-ms 180 --execute
```

Repeat separately for `tilt` and `zoom`, and for direction `-1`. Each pulse
produces a RANSAC match image plus a JSON record containing
`movement_detected`, `direction_correct`, and the measured response. The probe
does not change camera configuration automatically.

If ONVIF cannot provide the stream URI, pass the already tested RTSP URL with
`--stream-url`. Do not paste a URL containing credentials into bug reports.

## Practical limits

- The target must still overlap the current view enough for local features to
  match. Global search/exploration is not part of this first proof.
- Moving foliage, night/day transitions, snow, and large zoom changes reduce
  match quality. The controller refuses to move below its quality threshold.
- Cheap firmware may ignore PTZ speed. The duration and cumulative movement
  limits still bound the run, but start with conservative values.
- Pan/tilt sign conventions vary. The progress guard catches a wrong sign,
  but the dry run and inversion flags should be used first.

## Optional M1 stage two

Keep OpenCV as the portable baseline. A later optional backend can add:

- a small global image embedding for top-k place retrieval;
- ALIKED or DISK local features plus LightGlue for difficult illumination and
  viewpoint changes;
- PyTorch MPS acceleration when available, with CPU fallback;
- an empirical camera motion model keyed by zoom and movement direction.

These models should remain optional extras. The MVP file format and
`AlignmentResult` contract are intentionally backend-neutral so a learned
matcher can be added without coupling the PTZ safety controller to PyTorch.
