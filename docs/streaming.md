# Streaming Architecture

WatchMyBirds supports live camera streaming via two modes: **WebRTC relay** (through go2rtc) and **direct ffmpeg-based streaming**. The system automatically selects the best available mode.

---

## Streaming Modes

| Mode | Description | Latency |
|------|-------------|---------|
| **Relay (WebRTC)** | Camera stream relayed through go2rtc with WebRTC delivery to the browser | Low (~200–500 ms) |
| **Direct** | Camera stream decoded by ffmpeg and served as MJPEG to the browser | Higher (~1–3 s) |

### Mode Selection (`STREAM_SOURCE_MODE`)

| Value | Behavior |
|-------|----------|
| `auto` (default) | Use relay if go2rtc health check passes; fall back to direct |
| `relay` | Always use go2rtc relay URL |
| `direct` | Always use `CAMERA_URL` directly via ffmpeg |

---

## go2rtc Integration

WatchMyBirds manages go2rtc configuration automatically:

1. On startup, the app writes `go2rtc.yaml` with the user's `CAMERA_URL`
2. go2rtc reads this config and exposes the stream as an RTSP relay
3. The browser connects via WebRTC for low-latency playback

### Health Check

The app probes `GO2RTC_API_BASE` (default: `http://127.0.0.1:1984`) to determine relay availability.
If the probe fails, the app falls back to direct streaming automatically.

---

## Docker Networking

### Host Networking (Default)

The default `docker-compose.example.yml` uses **host networking** for both services:

```yaml
services:
  go2rtc:
    network_mode: host
  watchmybirds:
    network_mode: host
    environment:
      - GO2RTC_API_BASE=http://127.0.0.1:1984
```

This is the recommended setup because WebRTC requires direct UDP connectivity between the browser and go2rtc. Host networking provides this without additional STUN/TURN configuration.

### Bridge Networking (Alternative)

Bridge networking isolates containers but may prevent WebRTC from working without extra configuration. The app will automatically fall back to ffmpeg-based streaming in this case.

To use bridge networking:

1. Remove `network_mode: host` from both services
2. Add a shared Docker network
3. Map ports explicitly (`1984`, `8554`, `8555/tcp+udp`, `8050`)
4. Set `GO2RTC_API_BASE=http://go2rtc:1984` (service DNS)

See the comments in `docker-compose.example.yml` for a complete bridge configuration.

---

## RPi Appliance

On the Raspberry Pi appliance, both services run natively (no Docker):

- `app.service` — WatchMyBirds application on port 8050
- `go2rtc.service` — go2rtc relay on port 1984

The go2rtc config is written to `/opt/app/data/output/go2rtc.yaml` (writable under `ProtectSystem=strict`).

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_URL` | `""` | User-facing camera RTSP/HTTP URL |
| `STREAM_SOURCE_MODE` | `auto` | Source policy: `auto`, `relay`, `direct` |
| `GO2RTC_API_BASE` | `http://127.0.0.1:1984` | go2rtc health probe endpoint |
| `GO2RTC_CONFIG_PATH` | `./go2rtc.yaml` | Writable go2rtc config file path |
| `GO2RTC_STREAM_NAME` | `camera` | Stream name for `rtsp://<host>:8554/<name>` |
| `STREAM_FPS_CAPTURE` | `5.0` | Capture FPS throttle (reduces CPU load) |
| `STREAM_FPS` | `0` | UI MJPEG feed throttle (0 = unlimited) |
| `STREAM_WIDTH_OUTPUT_RESIZE` | `640` | Width for live stream preview in the UI |

---

## See Also

- [Configuration](CONFIGURATION.md) — Full configuration reference
- [Architecture](ARCHITECTURE.md) — System design overview
- [docker-compose.example.yml](../docker-compose.example.yml) — Docker deployment template
