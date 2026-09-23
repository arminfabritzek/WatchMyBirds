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
| D-Link DCS-6100LH | RTSP | 1080p | ✅ Working | Intermittent RTSP stalls reported before a stream-stability update | [@hmhaga](https://github.com/arminfabritzek/WatchMyBirds/issues/8#issuecomment-3883163802) | 2026-02 |
| SV3C PTZ (15× Optical Zoom) | RTSP | 2560×1920, 2560×1440 | ✅ Working | Advanced camera control was unreliable in testing; RTSP recommended | Developer | 2026-02 |
| Raspberry Pi Camera Module (CSI) | CSI / USB | 1080p | ✅ Working | Also works via motionEyeOS | Developer | 2026-02 |
| USB Webcam + motionEyeOS | RTSP | 1080p | ✅ Working | Stream via motionEyeOS on Raspberry Pi | Developer | 2026-02 |
| Blue Iris (NVR) | RTSP | up to 4K | ✅ Working | Acts as a camera proxy — any camera supported by Blue Iris works through its RTSP re-stream. | [@seglo](https://github.com/arminfabritzek/WatchMyBirds/pull/133) | 2026-09 |
| TP-Link Tapo C110 | RTSP | Not specified | ✅ Working | Docker; reporter adjusted lens focus | [@PabluskiNC](https://github.com/arminfabritzek/WatchMyBirds/issues/134) | 2026-09 |
| TP-Link Tapo C120 | RTSP | Not specified | ✅ Working | Docker; reporter adjusted lens focus | [@PabluskiNC](https://github.com/arminfabritzek/WatchMyBirds/issues/134) | 2026-09 |

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
