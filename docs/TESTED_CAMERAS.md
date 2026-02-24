# Tested Cameras

Community-verified cameras that work with WatchMyBirds.
If you have tested a camera not listed here, please open an issue or PR to add it!

## Legend

| Status | Meaning |
|--------|---------|
| ✅ Working | Fully functional — stream and detection confirmed |
| ⚠️ Partial | Stream works but with known limitations |
| ❌ Not Working | Could not establish a usable stream |

## Compatibility List

| Camera | Protocol | Resolution | Status | Notes | Reported by | Date |
|--------|----------|------------|--------|-------|-------------|------|
| D-Link DCS-6100LH | RTSP | 1080p | ✅ Working | — | Community | 2026-02 |
| SV3C PTZ (15× Optical Zoom) | RTSP | 2560×1920, 2560×1440 | ✅ Working | ONVIF supported but unreliable; RTSP recommended | Developer | 2026-02 |
| Raspberry Pi Camera Module (CSI) | CSI / USB | 1080p | ✅ Working | Also works via motionEyeOS | Developer | 2026-02 |
| USB Webcam + motionEyeOS | RTSP | 1080p | ✅ Working | Stream via motionEyeOS on Raspberry Pi | Developer | 2026-02 |

> **Note:** Cameras are tested by community members in their own environments.
> Results may vary depending on firmware version, network setup, and configuration.

## How to Report a Camera

Please include:

1. **Camera model** (full name)
2. **Protocol used** (RTSP, HTTP, USB)
3. **Resolution** tested at
4. **Status** (Working / Partial / Not Working)
5. **Any special notes** (e.g. firmware version, required settings)

Open an [issue](https://github.com/arminfabritzek/WatchMyBirds/issues) or submit a PR editing this file.
