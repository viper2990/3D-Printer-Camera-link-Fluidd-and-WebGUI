# Neptune 4 Plus Camera Monitor

Live camera stream + Discord notifications for your Klipper/Fluidd 3D printer.

---

## Code map — what each file does

### `app.py` — main application (Flask web server + camera + event watcher)

This is the entry point and glue layer. It has three major parts:

**`CameraService` class (lines 72–362)**
Owns the entire camera pipeline. A background thread (`_capture_loop`) reads frames from the USB camera using OpenCV, rotates them 180°, crops the edges (if configured), runs software stabilization (`_stabilize_frame`) to reduce vibration jitter, encodes each frame as a JPEG, and stores the latest one in memory. The web server reads that stored JPEG to serve the stream — the camera thread and the HTTP threads never block each other.

Key methods:
- `start()` / `stop()` — start or cleanly stop the background thread
- `_open_camera()` — negotiates resolution/FPS/MJPG mode with the V4L2 driver
- `_stabilize_frame()` — uses OpenCV phase correlation on a downscaled grayscale copy to measure and counteract small camera shake
- `_crop_frame()` — trims pixels from any edge (configured via env vars)
- `get_jpeg()` — thread-safe read of the latest encoded frame
- `get_status()` — returns a `StreamStatus` snapshot (online, FPS, resolution, alert, etc.)

**Event watcher thread (`_watch_events`, lines 496–647)**
Runs every 0.5 seconds. Polls Moonraker for printer state and sends Discord notifications for:
- Print started (with the slicer preview image from Klipper metadata)
- Progress milestones at 25 %, 50 %, 75 %, and 100 %
- Print failure (state goes to `error`) with a live camera snapshot
- Camera fault during an active print

State (which milestones were already sent, what the current filename is) is saved to `.milestones.json` so notifications survive an app restart mid-print.

**Flask routes (lines 654–820)**

| Route | Method | What it does |
|---|---|---|
| `/` | GET | Serves the dashboard HTML page |
| `/stream.mjpg` | GET | MJPEG stream (used by the web UI and Fluidd) |
| `/snapshot.jpg` | GET | Single JPEG snapshot of the latest frame |
| `/status` | GET | Camera health JSON (online, FPS, resolution, alert) |
| `/printer` | GET | Printer state JSON from Moonraker (state, progress, temps, etc.) |
| `/clear_alert` | POST | Dismisses the camera alert banner in the UI |
| `/settings` | GET/POST | Reads or saves Moonraker URL and Discord webhook to `.settings.json` |
| `/test_discord` | POST | Sends a test Discord ping |
| `/test_print_started` | POST | Sends a test "print started" Discord notification |

**Helper functions**
- `_format_time(seconds)` — converts seconds to a `3h 35m` style string
- `_fetch_preview_image(url)` — downloads the slicer thumbnail from Moonraker
- `_scale_image(bytes, scale)` — upscales an image (used to make the tiny slicer thumbnail bigger in Discord)
- `_send_notification(...)` — picks `notifier.send()` or `notifier.send_with_image()` automatically
- `_load_settings()` / `_save_settings()` — reads/writes `.settings.json`
- `_load_milestones_state()` / `_save_milestones_state()` — reads/writes `.milestones.json`

---

### `moonraker.py` — Klipper/Moonraker REST client

`MoonrakerClient` is a thin wrapper around the Moonraker HTTP API. It is only enabled when `MOONRAKER_URL` is set.

- `_get(path)` — basic GET with fail-soft error handling (returns `None` on any error)
- `_get_metadata(filename)` — fetches slicer metadata for the current file (one HTTP call, result shared)
- `_get_estimated_time(metadata)` — extracts estimated print time from metadata
- `_get_preview_url(metadata)` — extracts the slicer thumbnail URL from metadata
- `get_state()` — the main method called by `app.py`. Makes two HTTP calls per poll: one for live printer objects (state, progress, temps) and one for file metadata. Returns a flat dict with everything the dashboard and notifier need.

---

### `notifier.py` — Discord webhook sender

`Notifier` sends messages to a Discord channel via a webhook URL. It is only enabled when `DISCORD_WEBHOOK_URL` is set.

- `send(title, message, color, emoji)` — posts a plain embed (text only)
- `send_with_image(title, message, image_bytes, color, emoji)` — posts an embed with a JPEG attachment (used for milestone and failure snapshots)

---

### `templates/index.html` — web dashboard (single-page app)

All HTML, CSS, and JavaScript in one file. No framework or build step.

**Layout**: Three cards stacked vertically — page header, live camera stream + status badge, metrics grid + action buttons. A fourth settings card slides in when you click Settings.

**JavaScript polling**:
- `refreshStatus()` — calls `/status` every 2.5 s to update the camera state, FPS, resolution, dropped frames, and the live/alert badge overlay on the stream
- `refreshPrinter()` — calls `/printer` every 6 s to update printer state, filename, progress, and time remaining

**Buttons**:
- **Test Discord** — POST `/test_discord`
- **Test Print Started** — POST `/test_print_started`
- **Settings** — shows the settings card, loads current values from GET `/settings`
- **Dismiss Alert** — POST `/clear_alert`, hides the alert banner

---

### `start_monitor.sh` — launcher script

Sets all environment variables and starts `app.py` inside the local `.venv`. Run this instead of calling Python directly. Creates `.venv` and installs dependencies automatically on first run.

Accepts an optional argument for the camera device index (e.g. `./start_monitor.sh 1` for `/dev/video1`).

All tunable values are near the top — camera resolution, FPS, crop, stabilization settings, JPEG quality, stream FPS, Moonraker URL, and Discord webhook URL.

---

### `.settings.json` (runtime, not committed)

Persists the Moonraker URL and Discord webhook URL entered through the web Settings panel so they survive app restarts. Created automatically by the app.

### `.milestones.json` (runtime, not committed)

Persists which progress milestones (25 %, 50 %, 75 %, 100 %) have already been notified for the current print, and whether the "print started" notification was sent. Prevents duplicate Discord messages if the app restarts mid-print.

### `requirements.txt`

Three dependencies: `Flask` (web server), `opencv-python` (camera capture, image processing), `requests` (HTTP calls to Moonraker and Discord).

---

## Quick start

```bash
# 1. Find your camera index
ls /dev/video*

# 2. Set your printer IP and Discord webhook in start_monitor.sh, then run:
chmod +x start_monitor.sh
./start_monitor.sh          # camera 0
./start_monitor.sh 1        # camera 1

# 3. Open the dashboard
# http://<raspberry-pi-ip>:8080

# 4. In Fluidd, set the webcam URL to:
# http://<raspberry-pi-ip>:8080/stream.mjpg
```

## Key environment variables

| Variable | Default | Description |
|---|---|---|
| `MOONRAKER_URL` | `http://10.0.16.111:80` | Your printer's Moonraker address |
| `DISCORD_WEBHOOK_URL` | *(empty)* | Discord webhook for notifications |
| `CAMERA_INDEX` | `0` | `/dev/videoN` device number |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | `1280` / `720` | Capture resolution |
| `CAMERA_FPS` | `15` | Capture frame rate |
| `CAMERA_CROP_BOTTOM` | `270` | Pixels to trim from the bottom edge |
| `CAMERA_STABILIZE` | `true` | Software stabilization on/off |
| `STREAM_MJPEG_FPS` | `12` | Frame rate of the `/stream.mjpg` output |
| `MOONRAKER_POLL_INTERVAL_S` | `2.0` | How often to poll Moonraker for printer state |

All camera and stream options can be set as environment variables before running `start_monitor.sh`, or edited directly in that file.
