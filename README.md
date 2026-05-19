# Neptune 4 Plus Camera Monitor

Minimal camera bridge for Fluidd/Klipper with Discord notifications.

## 1) Check your camera device

```bash
ls /dev/video*
v4l2-ctl --list-devices
```

Find your C922 device index (`/dev/video0` => `CAMERA_INDEX=0`, etc).

If `v4l2-ctl` is missing:

```bash
sudo apt update && sudo apt install -y v4l-utils
```

## 2) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Run the server

```bash
export CAMERA_INDEX=0
export CAMERA_WIDTH=1280
export CAMERA_HEIGHT=720
export CAMERA_FPS=30
python app.py
```

Then open:

- http://localhost:8080

## Manual start file (no auto-start)

Use the included launcher script whenever you want to start the monitor:

```bash
chmod +x start_monitor.sh
./start_monitor.sh
```

If your camera is on a different index:

```bash
./start_monitor.sh 1
```

This only runs when you execute the file. It does not auto-start on boot.

## What this app does

- Exposes one low-latency camera stream at `/stream.mjpg`.
- Uses that same stream in the web UI and in Fluidd.
- Shows read-only Moonraker print status in the web page.
- Sends Discord alert notifications if the camera reports a fault during an active print.
- Supports manual Discord webhook test from the web page.

## Environment variables

Most users only need these:

```bash
MOONRAKER_URL=http://your-printer-ip:7125
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

Optional camera and stream tuning values are in `start_monitor.sh`.

## Removed by design

- No pause/resume/speed printer controls.
- No automatic pausing logic.
- No Discord bot polling commands.
