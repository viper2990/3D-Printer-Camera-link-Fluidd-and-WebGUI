import atexit
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

try:
    import cv2  # type: ignore
except ModuleNotFoundError:
    print("Error: Missing dependency 'opencv-python' (cv2).")
    print("Run from this folder using: source .venv/bin/activate && python app.py")
    print("Or install dependencies: pip install -r requirements.txt")
    sys.exit(1)

try:
    from flask import Flask, Response, jsonify, render_template, request
except ModuleNotFoundError:
    print("Error: Missing dependency 'Flask'.")
    print("Run from this folder using: source .venv/bin/activate && python app.py")
    print("Or install dependencies: pip install -r requirements.txt")
    sys.exit(1)

from detector import PrintAnomalyDetector
from moonraker import MoonrakerClient
from notifier import Notifier, TeamsNotifier

# Settings file for persisting configuration
_settings_file = os.path.join(os.path.dirname(__file__), ".settings.json")


def _load_settings() -> dict:
    try:
        if os.path.exists(_settings_file):
            with open(_settings_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        log.debug("Failed to load settings: %s", e)
    return {"moonraker_url": "", "discord_webhook": "", "teams_webhook": "",
            "crop_top": 0, "crop_bottom": 0, "crop_left": 0, "crop_right": 0}


def _save_settings(data: dict) -> None:
    try:
        with open(_settings_file, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        log.debug("Failed to save settings: %s", e)


@dataclass
class StreamStatus:
    # Live camera health and stream metrics shown in the UI.
    online: bool = False
    width: int = 0
    height: int = 0
    fps: float = 0.0
    alert: bool = False
    alert_reason: str = "No alert"
    dropped_frames: int = 0
    uptime_s: float = 0.0


class CameraService:
    # Owns capture, crop, stabilization, and JPEG encoding for the camera feed.
    def __init__(self) -> None:
        # Camera device and image settings come from environment variables.
        self.device_index = int(os.getenv("CAMERA_INDEX", "0"))
        self.target_width = int(os.getenv("CAMERA_WIDTH", "1280"))
        self.target_height = int(os.getenv("CAMERA_HEIGHT", "720"))
        self.target_fps = int(os.getenv("CAMERA_FPS", "30"))
        self.crop_top = max(int(os.getenv("CAMERA_CROP_TOP", "0")), 0)
        self.crop_bottom = max(int(os.getenv("CAMERA_CROP_BOTTOM", "0")), 0)
        self.crop_left = max(int(os.getenv("CAMERA_CROP_LEFT", "0")), 0)
        self.crop_right = max(int(os.getenv("CAMERA_CROP_RIGHT", "0")), 0)
        self.stabilize_enabled = os.getenv("CAMERA_STABILIZE", "true").lower() == "true"
        self.stabilize_scale = min(max(float(os.getenv("CAMERA_STABILIZE_SCALE", "0.25")), 0.1), 1.0)
        self.stabilize_smoothing = min(max(float(os.getenv("CAMERA_STABILIZE_SMOOTHING", "0.90")), 0.0), 0.99)
        self.stabilize_max_shift_px = max(float(os.getenv("CAMERA_STABILIZE_MAX_SHIFT", "8.0")), 1.0)
        self.stabilize_compute_every = max(int(os.getenv("CAMERA_STABILIZE_EVERY", "2")), 1)
        self.stabilize_deadband_px = max(float(os.getenv("CAMERA_STABILIZE_DEADBAND", "0.45")), 0.0)
        self.stabilize_preserve_fps = os.getenv("CAMERA_STABILIZE_PRESERVE_FPS", "true").lower() == "true"
        self.stabilize_min_fps = max(float(os.getenv("CAMERA_STABILIZE_MIN_FPS", "24")), 1.0)
        self.autofocus_enabled = os.getenv("CAMERA_AUTOFOCUS", "false").lower() == "true"
        self.focus_value = int(os.getenv("CAMERA_FOCUS", "0"))
        self.jpeg_quality = int(min(max(int(os.getenv("CAMERA_JPEG_QUALITY", "88")), 60), 100))
        self.jpeg_quality_floor = int(min(max(int(os.getenv("CAMERA_JPEG_QUALITY_FLOOR", "75")), 50), 100))
        self.jpeg_quality_preserve_fps = os.getenv("CAMERA_JPEG_PRESERVE_FPS", "true").lower() == "true"
        self.jpeg_quality_min_fps = max(float(os.getenv("CAMERA_JPEG_MIN_FPS", "24")), 1.0)

        self._cap: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        self._latest_jpeg: Optional[bytes] = None
        self._status = StreamStatus()

        self._started_at = time.time()
        self._frames_in_second = 0
        self._fps_clock = time.time()
        self._prev_stabilize_gray: Optional[Any] = None
        self._stabilize_dx = 0.0
        self._stabilize_dy = 0.0
        self._stabilize_frame_index = 0

    def start(self) -> None:
        # Start a background thread so the web server stays responsive.
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        # Cleanly stop the capture thread and release the USB camera.
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()

    def _apply_camera_controls(self, cap: Any) -> None:
        # Apply autofocus or fixed focus depending on the camera settings.
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1.0 if self.autofocus_enabled else 0.0)
        if not self.autofocus_enabled:
            cap.set(cv2.CAP_PROP_FOCUS, float(self.focus_value))
        log.info(
            "Camera controls: autofocus=%s, focus=%s",
            "on" if self.autofocus_enabled else "off",
            self.focus_value,
        )

    def _stabilize_frame(self, frame: Any) -> Any:
        # Reduce small camera shakes so the print area looks steadier.
        if not self.stabilize_enabled:
            return frame

        self._stabilize_frame_index += 1
        if self.stabilize_preserve_fps and self._status.fps > 0 and self._status.fps < self.stabilize_min_fps:
            # Prioritize frame rate: bypass stabilization work when FPS falls too low.
            self._prev_stabilize_gray = None
            self._stabilize_dx = 0.0
            self._stabilize_dy = 0.0
            return frame

        h, w = frame.shape[:2]

        def apply_offset(dx: float, dy: float) -> Any:
            transform = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, 1.0)
            transform[0, 2] -= dx
            transform[1, 2] -= dy
            return cv2.warpAffine(
                frame,
                transform,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

        should_compute = self.stabilize_compute_every <= 1 or (self._stabilize_frame_index % self.stabilize_compute_every) == 0
        if not should_compute:
            if abs(self._stabilize_dx) < self.stabilize_deadband_px and abs(self._stabilize_dy) < self.stabilize_deadband_px:
                return frame
            return apply_offset(self._stabilize_dx, self._stabilize_dy)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=self.stabilize_scale, fy=self.stabilize_scale, interpolation=cv2.INTER_AREA)

        if self._prev_stabilize_gray is None:
            self._prev_stabilize_gray = small
            return frame

        shift, _ = cv2.phaseCorrelate(
            self._prev_stabilize_gray.astype("float32"),
            small.astype("float32"),
        )
        self._prev_stabilize_gray = small

        dx, dy = shift
        scale_back = 1.0 / self.stabilize_scale
        dx *= scale_back
        dy *= scale_back

        dx = max(-self.stabilize_max_shift_px, min(self.stabilize_max_shift_px, dx))
        dy = max(-self.stabilize_max_shift_px, min(self.stabilize_max_shift_px, dy))

        if abs(dx) < self.stabilize_deadband_px and abs(dy) < self.stabilize_deadband_px:
            # Decay correction toward zero when motion is minimal.
            self._stabilize_dx *= self.stabilize_smoothing
            self._stabilize_dy *= self.stabilize_smoothing
            if abs(self._stabilize_dx) < self.stabilize_deadband_px and abs(self._stabilize_dy) < self.stabilize_deadband_px:
                return frame
            return apply_offset(self._stabilize_dx, self._stabilize_dy)

        s = self.stabilize_smoothing
        self._stabilize_dx = (s * self._stabilize_dx) + ((1.0 - s) * dx)
        self._stabilize_dy = (s * self._stabilize_dy) + ((1.0 - s) * dy)
        return apply_offset(self._stabilize_dx, self._stabilize_dy)

    def _crop_frame(self, frame: Any) -> Any:
        # Trim the edges if the camera sees extra dead space around the bed.
        if self.crop_top == 0 and self.crop_bottom == 0 and self.crop_left == 0 and self.crop_right == 0:
            return frame

        h, w = frame.shape[:2]
        y1 = min(self.crop_top, h - 1)
        y2 = max(y1 + 1, h - self.crop_bottom)
        x1 = min(self.crop_left, w - 1)
        x2 = max(x1 + 1, w - self.crop_right)

        return frame[y1:y2, x1:x2]

    def _open_camera(self) -> bool:
        # Open the camera and negotiate a usable video mode.
        if self._cap:
            self._cap.release()

        cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            self._status.online = False
            self._status.alert = True
            self._status.alert_reason = "Camera not available"
            return False

        # Keep the camera driver queue shallow to reduce old-frame latency.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Set size first, then force MJPG, then FPS. Some UVC drivers reset FOURCC
        # when width/height are changed, which can silently fall back to YUYV.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        fourcc = getattr(cv2, "VideoWriter_fourcc")(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._apply_camera_controls(cap)

        negotiated_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((negotiated_fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip()
        negotiated_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        negotiated_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        negotiated_fps = cap.get(cv2.CAP_PROP_FPS)
        log.info(
            "Camera negotiated mode: %sx%s %s @ %.1f fps",
            negotiated_w,
            negotiated_h,
            fourcc or "unknown",
            negotiated_fps,
        )
        log.info(
            "Camera stabilization: %s (scale=%.2f, smooth=%.2f, max_shift=%.1fpx, every=%s, preserve_fps=%s, min_fps=%.1f)",
            "on" if self.stabilize_enabled else "off",
            self.stabilize_scale,
            self.stabilize_smoothing,
            self.stabilize_max_shift_px,
            self.stabilize_compute_every,
            "on" if self.stabilize_preserve_fps else "off",
            self.stabilize_min_fps,
        )
        log.info(
            "Camera crop: top=%s, bottom=%s, left=%s, right=%s",
            self.crop_top,
            self.crop_bottom,
            self.crop_left,
            self.crop_right,
        )
        log.info(
            "JPEG quality: target=%s, floor=%s, preserve_fps=%s, min_fps=%.1f",
            self.jpeg_quality,
            self.jpeg_quality_floor,
            "on" if self.jpeg_quality_preserve_fps else "off",
            self.jpeg_quality_min_fps,
        )

        self._prev_stabilize_gray = None
        self._stabilize_dx = 0.0
        self._stabilize_dy = 0.0
        self._stabilize_frame_index = 0

        self._cap = cap
        self._status.online = True
        self._status.alert = False
        self._status.alert_reason = "No alert"
        return True

    def _capture_loop(self) -> None:
        # Main camera loop: read frames, process them, and store the latest JPEG.
        # Wait for USB devices to initialize on system boot
        boot_wait_s = 5
        log.info("Camera thread: waiting up to %d seconds for USB devices to initialize...", boot_wait_s)
        for _ in range(boot_wait_s):
            if not self._running:
                return
            time.sleep(1)
        
        while self._running:
            if not self._open_camera():
                time.sleep(1)
                continue

            while self._running and self._cap and self._cap.isOpened():
                ok, frame = self._cap.read()
                self._status.uptime_s = time.time() - self._started_at

                if not ok or frame is None:
                    self._status.dropped_frames += 1
                    self._status.online = False
                    self._status.alert = True
                    self._status.alert_reason = "Dropped camera frame"
                    break

                self._status.online = True

                # Rotate frame 180 degrees
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                frame = self._crop_frame(frame)
                frame = self._stabilize_frame(frame)

                self._status.width = int(frame.shape[1])
                self._status.height = int(frame.shape[0])

                encode_quality = self.jpeg_quality
                if (
                    self.jpeg_quality_preserve_fps
                    and self._status.fps > 0
                    and self._status.fps < self.jpeg_quality_min_fps
                ):
                    encode_quality = self.jpeg_quality_floor

                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), encode_quality],
                )
                if ok:
                    with self._lock:
                        self._latest_jpeg = encoded.tobytes()

                self._frames_in_second += 1
                now = time.time()
                if now - self._fps_clock >= 1.0:
                    elapsed = now - self._fps_clock
                    self._status.fps = self._frames_in_second / elapsed
                    self._fps_clock = now
                    self._frames_in_second = 0

    def get_jpeg(self) -> Optional[bytes]:
        # Return the newest encoded frame for streaming.
        with self._lock:
            return self._latest_jpeg

    def get_status(self) -> StreamStatus:
        # Return the latest camera health snapshot for the UI.
        return self._status


app = Flask(__name__)
# Shared services used by the web app and background workers.
camera = CameraService()
# Restore any crop values saved via the web UI.
_startup_settings = _load_settings()
if any(_startup_settings.get(k, 0) for k in ("crop_top", "crop_bottom", "crop_left", "crop_right")):
    camera.crop_top = _startup_settings.get("crop_top", 0)
    camera.crop_bottom = _startup_settings.get("crop_bottom", 0)
    camera.crop_left = _startup_settings.get("crop_left", 0)
    camera.crop_right = _startup_settings.get("crop_right", 0)
camera.start()
atexit.register(camera.stop)

moonraker = MoonrakerClient()
notifier = Notifier()
teams_notifier = TeamsNotifier()
detector = PrintAnomalyDetector()
_notify_fired = False
_moonraker_state_cache: dict[str, Any] = {}
_moonraker_last_fetch = 0.0
_moonraker_poll_interval_s = max(float(os.getenv("MOONRAKER_POLL_INTERVAL_S", "8.0")), 1.0)
_moonraker_lock = threading.Lock()
_previous_printer_state = "unknown"
_previous_progress = 0.0  # Track previous progress to detect skipped milestones
_current_print_filename = ""
_print_milestones_notified = set()  # Track which milestones (25, 50, 75, 100) have been notified
_failure_notified = False  # Track if we've notified about a spaghetti/blob failure
_notified_print_started_filename = ""  # Track which filename we've already sent "Print Started" for
_milestones_file = os.path.join(os.path.dirname(__file__), ".milestones.json")

# Live Discord countdown state
_live_discord_message_id: Optional[str] = None
_live_discord_milestone: int = 0
_live_discord_stop = threading.Event()
_live_discord_thread: Optional[threading.Thread] = None


def _load_milestones_state() -> dict:
    """Load milestone and print-start notification state from disk."""
    try:
        if os.path.exists(_milestones_file):
            with open(_milestones_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        log.debug("Failed to load milestones state: %s", e)
    return {"filename": "", "milestones": [], "notified_start_filename": ""}


def _save_milestones_state(filename: str, milestones: set, notified_start_filename: str = "") -> None:
    """Save milestone and print-start notification state to disk for persistence across restarts."""
    try:
        with open(_milestones_file, 'w') as f:
            json.dump({
                "filename": filename,
                "milestones": sorted(list(milestones)),
                "notified_start_filename": notified_start_filename
            }, f)
    except Exception as e:
        log.debug("Failed to save milestones state: %s", e)


# Load saved state on startup
_saved_state = _load_milestones_state()
_current_print_filename = _saved_state.get("filename", "")
_print_milestones_notified = set(_saved_state.get("milestones", []))
_notified_print_started_filename = _saved_state.get("notified_start_filename", "")


def _get_moonraker_state() -> dict[str, Any]:
    """Return Moonraker state. WebSocket: reads from in-process cache (free).
    REST fallback: rate-limited to avoid hammering the API."""
    global _moonraker_last_fetch, _moonraker_state_cache

    if not moonraker.enabled:
        return {}

    # WebSocket keeps an up-to-date cache in the client — read it directly.
    if moonraker.ws_enabled:
        return moonraker.get_state() or {}

    now = time.monotonic()
    with _moonraker_lock:
        if now - _moonraker_last_fetch >= _moonraker_poll_interval_s:
            _moonraker_state_cache = moonraker.get_state() or {}
            _moonraker_last_fetch = now
        return dict(_moonraker_state_cache)


def _format_time(seconds: int) -> str:
    """Format seconds into human-readable time string (e.g., '3h 35m', '15m')."""
    if seconds <= 0:
        return "0m"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"


def _fetch_preview_image(preview_url: str) -> Optional[bytes]:
    """Fetch preview image from Klipper metadata. Returns image bytes or None."""
    if not preview_url or not moonraker.enabled:
        return None
    try:
        full_url = moonraker.base_url + preview_url
        resp = requests.get(full_url, timeout=3)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        log.debug("Failed to fetch preview image: %s", e)
        return None


def _scale_image(image_bytes: Optional[bytes], scale: float = 2.0) -> Optional[bytes]:
    """Scale an image by the given factor. Returns scaled image bytes or None."""
    if not image_bytes:
        return None
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        height, width = img.shape[:2]
        scaled = cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LINEAR)
        _, encoded = cv2.imencode('.jpg', scaled)
        return encoded.tobytes()
    except Exception as e:
        log.debug("Failed to scale image: %s", e)
        return image_bytes


def _send_notification(title: str, message: str, image_bytes: Optional[bytes] = None,
                       color: int = 0x4db7ff, emoji: str = ":printer:",
                       facts: Optional[list] = None) -> Optional[str]:
    """Send to all channels. Returns the Discord message ID (for live edits), or None."""
    if image_bytes:
        msg_id = notifier.send_with_image(
            title=title, message=message, image_bytes=image_bytes,
            color=color, emoji=emoji, facts=facts,
        )
    else:
        msg_id = notifier.send(
            title=title, message=message, color=color, emoji=emoji, facts=facts,
        )
    teams_notifier.send(title=title, message=message, emoji=emoji, facts=facts)
    return msg_id


def _get_jpeg_wait(timeout: float = 10.0) -> Optional[bytes]:
    """Return a camera JPEG, waiting up to timeout seconds for the camera to warm up."""
    jpeg = camera.get_jpeg()
    if jpeg is not None:
        return jpeg
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.25)
        jpeg = camera.get_jpeg()
        if jpeg is not None:
            return jpeg
    return None


def _stop_live_discord_update() -> None:
    global _live_discord_thread
    _live_discord_stop.set()
    if _live_discord_thread and _live_discord_thread.is_alive():
        _live_discord_thread.join(timeout=3)
    _live_discord_thread = None


def _live_discord_loop(message_id: str, milestone: int, title: str, color: int) -> None:
    """Edit the milestone Discord message every 60 s with a fresh time remaining."""
    while not _live_discord_stop.wait(60):
        info = _get_moonraker_state()
        if not info or info.get("state") not in ("printing", "paused"):
            break
        progress = float(info.get("progress", milestone))
        remaining_s = int(info.get("remaining_s", 0))
        time_str = _format_time(remaining_s)
        filename = info.get("filename", _current_print_filename)
        notifier.edit_message(
            message_id=message_id,
            title=title,
            message="",
            color=color,
            emoji=":hourglass_flowing_sand:",
            facts=[
                {"title": "Progress", "value": f"{progress}%"},
                {"title": "Time Remaining", "value": time_str},
                {"title": "File", "value": filename},
            ],
        )


def _start_live_discord_update(message_id: str, milestone: int, title: str, color: int) -> None:
    global _live_discord_message_id, _live_discord_milestone, _live_discord_thread
    _stop_live_discord_update()
    _live_discord_message_id = message_id
    _live_discord_milestone = milestone
    _live_discord_stop.clear()
    t = threading.Thread(
        target=_live_discord_loop,
        args=(message_id, milestone, title, color),
        daemon=True,
    )
    _live_discord_thread = t
    t.start()
    log.info("Discord live countdown started for message %s", message_id)


def _watch_events() -> None:
    """Background thread: send Discord alerts for camera faults, print events, and failures."""
    global _notify_fired, _previous_printer_state, _previous_progress, _current_print_filename, _print_milestones_notified, _failure_notified, _notified_print_started_filename, detector, _live_discord_thread
    while True:
        # Poll frequently to catch 100% completion immediately (before state changes).
        time.sleep(0.5)

        # Treat paused prints as active because a failure can still be relevant.
        printer_state = "unknown"
        is_active_print = False
        print_info = {}
        if moonraker.enabled:
            info = _get_moonraker_state()
            printer_state = info.get("state", "unknown")
            is_active_print = printer_state in ("printing", "paused")
            print_info = info
            
            # Log state changes for debugging
            if printer_state != _previous_printer_state:
                log.info("Printer state changed: %s -> %s", _previous_printer_state, printer_state)

        # Check for print start
        if _previous_printer_state != "printing" and printer_state == "printing":
            filename = print_info.get("filename", "Unknown")
            if _current_print_filename != filename:
                _current_print_filename = filename
                _print_milestones_notified.clear()
                _failure_notified = False
                _save_milestones_state("", _print_milestones_notified, "")
                detector.reset()
                log.info("Print started: %s", filename)
            
            # Send print start notification only once per unique filename
            if _notified_print_started_filename != filename:
                _notified_print_started_filename = filename
                
                estimated_s = print_info.get("estimated_s", 0)
                time_str = _format_time(estimated_s) if estimated_s > 0 else ""
                message = f"Print started: {filename}"
                if time_str:
                    message += f"\nEstimated time: {time_str}"

                teams_facts = [{"title": "File", "value": filename}]
                if time_str:
                    teams_facts.append({"title": "Estimated Time", "value": time_str})

                # Send with preview image from Klipper metadata
                preview_url = print_info.get("preview_url", "")
                preview_image = _fetch_preview_image(preview_url)
                # Scale preview image 3x for better visibility in Discord
                preview_image = _scale_image(preview_image, scale=3.0)

                _send_notification(
                    title="Print Started",
                    message=message,
                    image_bytes=preview_image,
                    color=0x00FF00,
                    emoji=":printer:",
                    facts=teams_facts,
                )
                # Persist print start notification state to survive app restarts
                _save_milestones_state(_current_print_filename, _print_milestones_notified, filename)

        # Check for print failure (spaghetti/blob)
        if _previous_printer_state in ("printing", "paused") and printer_state == "error":
            if not _failure_notified:
                progress = float(print_info.get("progress", 0))
                log.warning("Print failed at %.1f%%: %s (possible spaghetti/blob/jam)", progress, _current_print_filename)
                message = f"Print failed at {progress}%: {_current_print_filename}\n\nPossible issues: spaghetti, blob, or filament jam."
                
                camera_jpeg = _get_jpeg_wait()
                _send_notification(
                    title="Print Failed!",
                    message=message,
                    image_bytes=camera_jpeg,
                    color=0xFF4444,
                    emoji=":rotating_light:"
                )
                _failure_notified = True

        # Check for print progress milestones
        if is_active_print and printer_state in ("printing", "paused"):
            progress = float(print_info.get("progress", 0))
            remaining_s = int(print_info.get("remaining_s", 0))
            
            for milestone in [25, 50, 75, 100]:
                if _previous_progress < milestone <= progress and milestone not in _print_milestones_notified:
                    log.info("Print milestone reached: %d%% (progress jumped from %.1f%% to %.1f%%)", 
                             milestone, _previous_progress, progress)
                    time_str = _format_time(remaining_s)
                    
                    if milestone == 100:
                        _stop_live_discord_update()
                        message = f"Print completed: {_current_print_filename}"
                        emoji = ":tada:"
                        color = 0x00FF00
                        notif_title = "Print Complete"
                        facts = [{"title": "File", "value": _current_print_filename}]
                    else:
                        _stop_live_discord_update()
                        message = ""
                        emoji = ":hourglass_flowing_sand:"
                        color = 0x4db7ff
                        notif_title = f"Print {milestone}% Complete"
                        facts = [
                            {"title": "Progress", "value": f"{milestone}%"},
                            {"title": "Time Remaining", "value": time_str},
                            {"title": "File", "value": _current_print_filename},
                        ]

                    # 100%: grab immediately before the head homes up and moves the camera.
                    # Other milestones: wait up to 10 s in case the app just restarted.
                    camera_jpeg = camera.get_jpeg() if milestone == 100 else _get_jpeg_wait()
                    msg_id = _send_notification(
                        title=notif_title,
                        message=message,
                        image_bytes=camera_jpeg,
                        color=color,
                        emoji=emoji,
                        facts=facts,
                    )
                    # Start live countdown for 25/50/75 — not needed for 100 (print done).
                    if milestone != 100 and msg_id:
                        _start_live_discord_update(msg_id, milestone, notif_title, color)

                    _print_milestones_notified.add(milestone)
                    _save_milestones_state(_current_print_filename, _print_milestones_notified, _notified_print_started_filename)
        # Spaghetti and blob detection — runs every DETECTOR_INTERVAL_S during active prints.
        if is_active_print and detector.should_check():
            jpeg = camera.get_jpeg()
            if jpeg:
                spaghetti, blob = detector.analyze(jpeg)
                progress = print_info.get("progress", 0)
                if spaghetti:
                    _send_notification(
                        title="Spaghetti Detected!",
                        message=f"Possible spaghetti detected.\nFile: {_current_print_filename}\nProgress: {progress}%",
                        image_bytes=jpeg,
                        color=0xFF8C00,
                        emoji=":warning:",
                    )
                if blob:
                    _send_notification(
                        title="Blob Detected!",
                        message=f"Possible filament blob detected.\nFile: {_current_print_filename}\nProgress: {progress}%",
                        image_bytes=jpeg,
                        color=0xFF4400,
                        emoji=":warning:",
                    )

        elif printer_state not in ("printing", "paused"):
            # Only act on the transition printing → idle, not on every idle tick.
            if _previous_printer_state in ("printing", "paused"):
                # Catch 100% if the state changed before the milestone loop saw it.
                if 100 not in _print_milestones_notified:
                    progress = float(print_info.get("progress", 0))
                    if progress >= 100.0:
                        log.info("Print completed at 100%% (state transition: %s -> %s)", _previous_printer_state, printer_state)
                        camera_jpeg = camera.get_jpeg()
                        _send_notification(
                            title="Print Complete",
                            message=f"Print completed: {_current_print_filename}",
                            image_bytes=camera_jpeg,
                            color=0x00FF00,
                            emoji=":tada:"
                        )
                # Stop any live Discord countdown and reset all print state.
                _stop_live_discord_update()
                _print_milestones_notified.clear()
                _current_print_filename = ""
                _failure_notified = False
                _notified_print_started_filename = ""
                _save_milestones_state("", _print_milestones_notified, "")
                _previous_progress = 0.0
                detector.reset()

        # Update progress tracker and state for next poll cycle
        if is_active_print:
            _previous_progress = float(print_info.get("progress", 0))
        
        _previous_printer_state = printer_state

        # Only notify once per alert streak.
        s = camera.get_status()
        if s.alert and is_active_print:
            if not _notify_fired:
                _send_notification(
                    title="Camera Alert During Print",
                    message=(
                        f"Camera issue detected: {s.alert_reason}. "
                        f"Printer state: {printer_state}."
                    ),
                )
                _notify_fired = True
        else:
            _notify_fired = False


_watcher = threading.Thread(target=_watch_events, daemon=True)
_watcher.start()


# -------------------- Routes --------------------
@app.route("/")
def index():
    # Main dashboard page.
    return render_template("index.html")


@app.route("/stream.mjpg")
def stream_mjpg():
    # Raw MJPEG stream for Fluidd fallback and direct browser viewing.
    stream_fps = max(float(os.getenv("STREAM_MJPEG_FPS", "8")), 1.0)
    min_interval = 1.0 / stream_fps

    def generate():
        # Keep yielding frames forever while the browser/client stays connected.
        last_sent = 0.0
        while True:
            frame = camera.get_jpeg()
            if frame is None:
                # Wait briefly until the camera has produced the first frame.
                time.sleep(0.05)
                continue

            # Throttle output so we do not exceed the configured stream FPS.
            now = time.time()
            elapsed = now - last_sent
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_sent = time.time()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    # Camera health and alert status for the frontend.
    s = camera.get_status()
    alert = s.alert
    alert_reason = s.alert_reason

    if moonraker.enabled:
        # Hide camera alerts when the printer is idle to reduce noise.
        state = _get_moonraker_state().get("state", "")
        if state not in ("printing", "paused"):
            alert = False
            alert_reason = "No active print"

    return jsonify(
        {
            "online": s.online,
            "width": s.width,
            "height": s.height,
            "fps": round(s.fps, 1),
            "alert": alert,
            "alert_reason": alert_reason,
            "dropped_frames": s.dropped_frames,
            "uptime_s": int(s.uptime_s),
            "moonraker_enabled": moonraker.enabled,
        }
    )


@app.route("/printer")
def printer():
    """Return current Klipper/Moonraker printer state."""
    # Read-only printer state for the dashboard.
    state = _get_moonraker_state()
    if not state:
        return jsonify({"connected": False})
    state["connected"] = True
    return jsonify(state)


@app.route("/clear_alert", methods=["POST"])
def clear_alert():
    """Dismiss a camera alert in the local UI and reset alert notifications."""
    # Only resets the local flag so the banner can clear in the browser.
    global _notify_fired
    s = camera.get_status()
    s.alert = False
    s.alert_reason = "Alert cleared by user"
    _notify_fired = False
    return jsonify({"ok": True})


@app.route("/settings")
def get_settings():
    settings = _load_settings()
    # Include live crop values from the running camera instance.
    settings["crop_top"] = camera.crop_top
    settings["crop_bottom"] = camera.crop_bottom
    settings["crop_left"] = camera.crop_left
    settings["crop_right"] = camera.crop_right
    return jsonify(settings)


@app.route("/crop", methods=["POST"])
def set_crop():
    """Apply crop values to the running camera immediately — no restart needed."""
    try:
        data = request.get_json() or {}
        camera.crop_top = max(int(data.get("crop_top", 0)), 0)
        camera.crop_bottom = max(int(data.get("crop_bottom", 0)), 0)
        camera.crop_left = max(int(data.get("crop_left", 0)), 0)
        camera.crop_right = max(int(data.get("crop_right", 0)), 0)
        log.info("Crop updated: top=%d bottom=%d left=%d right=%d",
                 camera.crop_top, camera.crop_bottom, camera.crop_left, camera.crop_right)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "reason": str(e)}), 400


@app.route("/settings", methods=["POST"])
def save_settings():
    try:
        data = request.get_json() or {}
        moonraker_url = data.get("moonraker_url", "").strip()
        discord_webhook = data.get("discord_webhook", "").strip()
        teams_webhook = data.get("teams_webhook", "").strip()

        crop_top = max(int(data.get("crop_top", 0)), 0)
        crop_bottom = max(int(data.get("crop_bottom", 0)), 0)
        crop_left = max(int(data.get("crop_left", 0)), 0)
        crop_right = max(int(data.get("crop_right", 0)), 0)

        _save_settings({
            "moonraker_url": moonraker_url,
            "discord_webhook": discord_webhook,
            "teams_webhook": teams_webhook,
            "crop_top": crop_top,
            "crop_bottom": crop_bottom,
            "crop_left": crop_left,
            "crop_right": crop_right,
        })

        # Apply changes immediately — no restart needed.
        if discord_webhook:
            notifier.webhook_url = discord_webhook
            notifier.enabled = True
        if teams_webhook:
            teams_notifier.webhook_url = teams_webhook
            teams_notifier.enabled = True
        camera.crop_top = crop_top
        camera.crop_bottom = crop_bottom
        camera.crop_left = crop_left
        camera.crop_right = crop_right

        log.info("Settings saved from web UI")
        return jsonify({"ok": True, "reason": "Settings saved"})
    except Exception as e:
        log.warning("Failed to save settings: %s", e)
        return jsonify({"ok": False, "reason": str(e)}), 400


@app.route("/test_discord", methods=["POST"])
def test_discord():
    """Send a manual Discord test notification from the web UI."""
    # Useful to confirm the webhook is wired correctly.
    if not notifier.enabled:
        return jsonify({"ok": False, "reason": "Discord webhook not configured"}), 503

    msg_id = notifier.send(
        title="Discord Test",
        message="Manual test alert from Neptune 4 Plus monitor UI.",
    )
    if msg_id is None:
        return jsonify({"ok": False, "reason": "Discord notification failed"}), 502
    return jsonify({"ok": True})


@app.route("/test_teams", methods=["POST"])
def test_teams():
    """Send a manual Teams test notification from the web UI."""
    if not teams_notifier.enabled:
        return jsonify({"ok": False, "reason": "Teams webhook not configured — add it in Settings"}), 503
    ok = teams_notifier.send(
        title="Teams Test",
        message="Manual test alert from Neptune 4 Plus monitor UI.",
        emoji=":printer:",
    )
    if not ok:
        return jsonify({"ok": False, "reason": "Teams notification failed"}), 502
    return jsonify({"ok": True})


@app.route("/test_print_started", methods=["POST"])
def test_print_started():
    """Send a test print started notification using the current print's real data."""
    if not notifier.enabled:
        return jsonify({"ok": False, "reason": "Discord webhook not configured"}), 503

    print_info = moonraker.get_state() or {}
    filename = print_info.get("filename") or "test_print.gcode"
    estimated_s = print_info.get("estimated_s", 5400)
    time_str = _format_time(estimated_s) if estimated_s > 0 else ""

    message = f"Print started: {filename}"
    if time_str:
        message += f"\nEstimated time: {time_str}"

    teams_facts = [{"title": "File", "value": filename}]
    if time_str:
        teams_facts.append({"title": "Estimated Time", "value": time_str})

    preview_url = print_info.get("preview_url", "")
    preview_image = _fetch_preview_image(preview_url)
    preview_image = _scale_image(preview_image, scale=3.0)

    msg_id = _send_notification(
        title="Print Started",
        message=message,
        image_bytes=preview_image,
        color=0x00FF00,
        emoji=":printer:",
        facts=teams_facts,
    )
    if msg_id is None and not preview_image:
        return jsonify({"ok": False, "reason": "Notification failed or no preview image available"}), 502
    return jsonify({"ok": True, "has_image": preview_image is not None})


@app.route("/snapshot.jpg")
def snapshot():
    # Single-frame JPEG snapshot for debugging or webhook attachments.
    frame = camera.get_jpeg()
    if frame is None:
        return ("No frame available yet", 503)
    return Response(frame, mimetype="image/jpeg")


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    app.run(host=host, port=port, debug=False, threaded=True)
