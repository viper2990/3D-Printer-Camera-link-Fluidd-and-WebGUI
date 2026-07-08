#!/usr/bin/env bash
set -euo pipefail

# Always run from the project directory.
cd "$(dirname "$0")"

if [[ ! -d ".venv" ]]; then
  echo "Creating local virtual environment (.venv)..."
  /usr/bin/python3 -m venv .venv
fi

source .venv/bin/activate

if ! python -c "import cv2, flask" >/dev/null 2>&1; then
  # Install the Python dependencies into the local virtual environment.
  echo "Installing Python dependencies..."
  pip install -r requirements.txt
fi

# Optional: pass camera index as first arg, default 0.
CAMERA_INDEX="${1:-0}"

# Camera capture settings.
export CAMERA_INDEX
export CAMERA_WIDTH="${CAMERA_WIDTH:-1280}"
export CAMERA_HEIGHT="${CAMERA_HEIGHT:-720}"
export CAMERA_FPS="${CAMERA_FPS:-15}"
export CAMERA_CROP_TOP="${CAMERA_CROP_TOP:-0}"
export CAMERA_CROP_BOTTOM="${CAMERA_CROP_BOTTOM:-0}"
export CAMERA_CROP_LEFT="${CAMERA_CROP_LEFT:-0}"
export CAMERA_CROP_RIGHT="${CAMERA_CROP_RIGHT:-0}"
export CAMERA_STABILIZE="${CAMERA_STABILIZE:-true}"
export CAMERA_STABILIZE_SCALE="${CAMERA_STABILIZE_SCALE:-0.25}"
export CAMERA_STABILIZE_SMOOTHING="${CAMERA_STABILIZE_SMOOTHING:-0.90}"
export CAMERA_STABILIZE_MAX_SHIFT="${CAMERA_STABILIZE_MAX_SHIFT:-8}"
export CAMERA_STABILIZE_EVERY="${CAMERA_STABILIZE_EVERY:-2}"
export CAMERA_STABILIZE_DEADBAND="${CAMERA_STABILIZE_DEADBAND:-0.45}"
export CAMERA_STABILIZE_PRESERVE_FPS="${CAMERA_STABILIZE_PRESERVE_FPS:-true}"
export CAMERA_STABILIZE_MIN_FPS="${CAMERA_STABILIZE_MIN_FPS:-24}"
export CAMERA_AUTOFOCUS="${CAMERA_AUTOFOCUS:-false}"
export CAMERA_FOCUS="${CAMERA_FOCUS:-0}"
export CAMERA_JPEG_QUALITY="${CAMERA_JPEG_QUALITY:-70}"
export CAMERA_JPEG_QUALITY_FLOOR="${CAMERA_JPEG_QUALITY_FLOOR:-60}"
export CAMERA_JPEG_PRESERVE_FPS="${CAMERA_JPEG_PRESERVE_FPS:-true}"
export CAMERA_JPEG_MIN_FPS="${CAMERA_JPEG_MIN_FPS:-24}"

# Browser stream settings.
# These tune the direct MJPEG stream used by the web UI and Fluidd.
export STREAM_MJPEG_FPS="${STREAM_MJPEG_FPS:-12}"

# Moonraker printer status endpoint.
# Set to your printer's IP to enable Klipper integration, e.g.:
# export MOONRAKER_URL="http://192.168.1.100:7125"
export MOONRAKER_URL="${MOONRAKER_URL:-http://10.0.16.111:80}"
export MOONRAKER_POLL_INTERVAL_S="${MOONRAKER_POLL_INTERVAL_S:-2.0}"

# Discord alert webhook.
# Discord webhook URL for fail notifications.
# Create one in Discord: Channel Settings -> Integrations -> Webhooks -> New Webhook -> Copy URL
export DISCORD_WEBHOOK_URL="${DISCORD_WEBHOOK_URL:-https://discord.com/api/webhooks/1502152505944178688/s9MUfI43ycBhhuGgUbDqocA4pkFsvzDYiR_MGQx27jnlOT8A-G-cVpx40mL6285poWk4}"

# Microsoft Teams alert webhook.
# How to set up:
#   1. Open your Teams channel → click + next to channel name → search "Workflows"
#   2. Choose "Post to a channel when a webhook request is received"
#   3. Follow the prompts — copy the webhook URL at the end and paste it below.
# Note: Teams notifications are text-only (no camera image attached).
export TEAMS_WEBHOOK_URL="${TEAMS_WEBHOOK_URL:-}"

# Spaghetti / blob detection.
# Uncomment and adjust these if you get false positives or missed detections.
# export DETECTOR_ENABLED=true
# export DETECTOR_INTERVAL_S=30          # how often to sample a frame (seconds)
# export DETECTOR_WARMUP=5              # checks before the baseline is established
# export DETECTOR_CONSECUTIVE=2         # consecutive hits before an alert fires
# export DETECTOR_COOLDOWN_S=300        # minimum seconds between repeat alerts
# export DETECTOR_SPAGHETTI_MULTIPLIER=2.0  # raise this to reduce spaghetti false positives
# export DETECTOR_BLOB_AREA_MIN=0.06        # raise this if small blobs trigger false alerts
# export DETECTOR_ROI_START=0.4            # 0.0=full frame, 0.6=bottom 40% only

echo "Starting monitor with CAMERA_INDEX=${CAMERA_INDEX}"
echo "Open: http://127.0.0.1:8080"
python app.py
