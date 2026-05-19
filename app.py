import atexit
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

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
    from flask import Flask, Response, jsonify, render_template
except ModuleNotFoundError:
    print("Error: Missing dependency 'Flask'.")
    print("Run from this folder using: source .venv/bin/activate && python app.py")
    print("Or install dependencies: pip install -r requirements.txt")
    sys.exit(1)

from moonraker import MoonrakerClient
from notifier import Notifier


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
        boot_wait_attempts = 5
        log.info("Camera thread: waiting up to 5 seconds for USB devices to initialize...")
        for attempt in range(boot_wait_attempts):
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
camera.start()
atexit.register(camera.stop)

moonraker = MoonrakerClient()
notifier = Notifier()
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
_milestones_file = os.path.join(os.path.dirname(__file__), ".milestones.json")


def _load_milestones_state() -> dict:
    """Load persistent milestone state from disk."""
    try:
        if os.path.exists(_milestones_file):
            with open(_milestones_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        log.debug("Failed to load milestones state: %s", e)
    return {"filename": "", "milestones": []}


def _save_milestones_state(filename: str, milestones: set) -> None:
    """Save milestone state to disk for persistence across restarts."""
    try:
        with open(_milestones_file, 'w') as f:
            json.dump({"filename": filename, "milestones": sorted(list(milestones))}, f)
    except Exception as e:
        log.debug("Failed to save milestones state: %s", e)


# Load saved state on startup
_saved_state = _load_milestones_state()
_current_print_filename = _saved_state.get("filename", "")
_print_milestones_notified = set(_saved_state.get("milestones", []))


def _get_moonraker_state() -> dict[str, Any]:
    """Return cached Moonraker state and refresh at a controlled interval."""
    global _moonraker_last_fetch, _moonraker_state_cache

    if not moonraker.enabled:
        return {}

    now = time.monotonic()
    with _moonraker_lock:
        if now - _moonraker_last_fetch >= _moonraker_poll_interval_s:
            _moonraker_state_cache = moonraker.get_state() or {}
            _moonraker_last_fetch = now
        return dict(_moonraker_state_cache)


def _watch_events() -> None:
    """Background thread: send Discord alerts for camera faults, print events, and failures."""
    global _notify_fired, _previous_printer_state, _previous_progress, _current_print_filename, _print_milestones_notified, _failure_notified
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

        # Check for print start (or app restart during a print)
        if (_previous_printer_state != "printing" and printer_state == "printing") or \
           (is_active_print and _current_print_filename != print_info.get("filename", "")):
            filename = print_info.get("filename", "Unknown")
            # Only log if it's a different print or a fresh start
            if _current_print_filename != filename:
                _current_print_filename = filename
                _print_milestones_notified.clear()
                _failure_notified = False
                _save_milestones_state(filename, _print_milestones_notified)
                log.info("Print started: %s", filename)
                
                estimated_s = print_info.get("estimated_s", 0)
                if estimated_s > 0:
                    hours = estimated_s // 3600
                    minutes = (estimated_s % 3600) // 60
                    time_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
                    message = f"Print started: {filename}\nEstimated time: {time_str}"
                else:
                    message = f"Print started: {filename}"
                
                notifier.send(
                    title="Print Started",
                    message=message,
                    color=0x00FF00,
                    emoji=":printer:"
                )

        # Check for print failure (spaghetti/blob)
        if _previous_printer_state in ("printing", "paused") and printer_state == "error":
            if not _failure_notified:
                progress = float(print_info.get("progress", 0))
                log.warning("Print failed at %.1f%%: %s (possible spaghetti/blob/jam)", progress, _current_print_filename)
                message = f"Print failed at {progress}%: {_current_print_filename}\n\nPossible issues: spaghetti, blob, or filament jam."
                
                notifier.send(
                    title="Print Failed!",
                    message=message,
                    color=0xFF4444,
                    emoji=":rotating_light:"
                )
                _failure_notified = True

        # Check for print progress milestones
        # Continuous milestone checking - catch any missed milestones during normal operation
        if is_active_print and printer_state in ("printing", "paused"):
            progress = float(print_info.get("progress", 0))
            remaining_s = int(print_info.get("remaining_s", 0))
            
            # Check milestones: 25%, 50%, 75%, 100%
            for milestone in [25, 50, 75, 100]:
                if _previous_progress < milestone <= progress and milestone not in _print_milestones_notified:
                    log.info("Print milestone reached: %d%% (progress jumped from %.1f%% to %.1f%%)", milestone, _previous_progress, progress)
                    hours = remaining_s // 3600
                    minutes = (remaining_s % 3600) // 60
                    time_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
                    
                    if milestone == 100:
                        message = f"Print completed: {_current_print_filename}"
                        emoji = ":tada:"
                        color = 0x00FF00
                    else:
                        message = f"Print progress: {milestone}%\nTime remaining: {time_str}"
                        emoji = ":hourglass_flowing_sand:"
                        color = 0x4db7ff
                    
                    notifier.send(
                        title=f"Print {milestone}% Complete" if milestone != 100 else "Print Complete",
                        message=message,
                        color=color,
                        emoji=emoji
                    )
                    _print_milestones_notified.add(milestone)
                    _save_milestones_state(_current_print_filename, _print_milestones_notified)
        elif printer_state not in ("printing", "paused"):
            # Check if 100% was missed when transitioning out of printing state
            if _previous_printer_state in ("printing", "paused") and 100 not in _print_milestones_notified:
                progress = float(print_info.get("progress", 0))
                if progress >= 100.0:
                    log.info("Print completed at 100%% (state transition: %s -> %s)", _previous_printer_state, printer_state)
                    notifier.send(
                        title="Print Complete",
                        message=f"Print completed: {_current_print_filename}",
                        color=0x00FF00,
                        emoji=":tada:"
                    )
                    _print_milestones_notified.add(100)
            
            # Reset milestones when print ends
            _print_milestones_notified.clear()
            _current_print_filename = ""
            _failure_notified = False
            _save_milestones_state("", _print_milestones_notified)
            _previous_progress = 0.0

        # Update progress tracker and state for next poll cycle
        if is_active_print:
            _previous_progress = float(print_info.get("progress", 0))
        
        _previous_printer_state = printer_state

        # Only notify once per alert streak.
        s = camera.get_status()
        if s.alert and is_active_print:
            # Send one Discord alert per alert event so the channel does not spam.
            if not _notify_fired:
                notifier.send(
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


@app.route("/test_discord", methods=["POST"])
def test_discord():
    """Send a manual Discord test notification from the web UI."""
    # Useful to confirm the webhook is wired correctly.
    if not notifier.enabled:
        return jsonify({"ok": False, "reason": "Discord webhook not configured"}), 503

    ok = notifier.send(
        title="Discord Test",
        message="Manual test alert from Neptune 4 Plus monitor UI.",
    )
    if not ok:
        return jsonify({"ok": False, "reason": "Discord notification failed"}), 502
    return jsonify({"ok": True})


@app.route("/test_print_started", methods=["POST"])
def test_print_started():
    """Send a test print started notification to verify functionality."""
    if not notifier.enabled:
        return jsonify({"ok": False, "reason": "Discord webhook not configured"}), 503

    # Simulate a print started notification with sample data
    filename = "test_print.gcode"
    estimated_s = 5400  # 1.5 hours in seconds
    hours = estimated_s // 3600
    minutes = (estimated_s % 3600) // 60
    time_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
    message = f"Print started: {filename}\nEstimated time: {time_str}"
    
    ok = notifier.send(
        title="Print Started",
        message=message,
        color=0x00FF00,
        emoji=":printer:"
    )
    if not ok:
        return jsonify({"ok": False, "reason": "Print started notification failed"}), 502
    return jsonify({"ok": True})


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
