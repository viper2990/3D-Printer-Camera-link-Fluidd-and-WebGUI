import json
import logging
import os
import threading
import time
from typing import Any, Optional
from urllib.parse import quote, urlparse

log = logging.getLogger(__name__)

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ModuleNotFoundError:
    _HAS_REQUESTS = False
    _requests = None  # type: ignore

try:
    import websocket as _websocket
    _HAS_WEBSOCKET = True
except ModuleNotFoundError:
    _HAS_WEBSOCKET = False
    _websocket = None  # type: ignore


class MoonrakerClient:
    """
    Moonraker client with WebSocket as primary connection and REST as fallback.

    WebSocket: connects to ws://<host>/websocket, subscribes to printer objects,
    and receives incremental state pushes from Moonraker in real time.
    Install websocket-client (already in requirements.txt) to enable.

    REST fallback: used automatically if websocket-client is missing or the
    WebSocket connection has not yet established.
    """

    def __init__(self) -> None:
        self.base_url = os.getenv("MOONRAKER_URL", "").rstrip("/")
        self._timeout = 3
        self.enabled = bool(self.base_url and _HAS_REQUESTS)
        self.ws_enabled = False  # flips True once the WebSocket subscribes successfully

        self._ws_url = self._build_ws_url()
        self._state_cache: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._rpc_id = 0
        self._metadata_cache: dict[str, dict] = {}  # keyed by filename

        if self.base_url and _HAS_WEBSOCKET:
            log.info("Moonraker WebSocket connecting: %s", self._ws_url)
            t = threading.Thread(target=self._ws_loop, daemon=True)
            t.start()
        elif self.enabled:
            log.info("Moonraker REST polling active (install websocket-client for WebSocket): %s", self.base_url)
        else:
            log.info("Moonraker not configured. Set MOONRAKER_URL to enable.")

    # ------------------------------------------------------------------
    # WebSocket layer
    # ------------------------------------------------------------------

    def _build_ws_url(self) -> str:
        if not self.base_url:
            return ""
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return f"{scheme}://{parsed.netloc}/websocket"

    def _next_id(self) -> int:
        self._rpc_id += 1
        return self._rpc_id

    def _ws_loop(self) -> None:
        retry_delay = 2.0
        while True:
            try:
                self._connect_and_run()
                retry_delay = 2.0
            except Exception as exc:
                log.debug("Moonraker WebSocket error: %s", exc)
            with self._lock:
                self.ws_enabled = False
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 1.5, 30.0)

    def _connect_and_run(self) -> None:
        ws = _websocket.WebSocket()  # type: ignore[union-attr]
        ws.connect(self._ws_url, timeout=10)

        ws.send(json.dumps({
            "jsonrpc": "2.0",
            "method": "printer.objects.subscribe",
            "params": {
                "objects": {
                    "print_stats": None,
                    "display_status": None,
                    "heater_bed": None,
                    "extruder": None,
                }
            },
            "id": self._next_id(),
        }))

        while True:
            raw = ws.recv()
            if not raw:
                break
            self._handle_message(json.loads(raw))

    def _handle_message(self, msg: dict) -> None:
        # Subscription response — contains the full initial state snapshot.
        if "result" in msg and isinstance(msg["result"], dict) and "status" in msg["result"]:
            with self._lock:
                self._merge(msg["result"]["status"])
                self.ws_enabled = True
            log.info("Moonraker WebSocket subscribed — real-time updates active")
            return

        method = msg.get("method", "")
        if method == "notify_status_update":
            params = msg.get("params", [])
            if params and isinstance(params[0], dict):
                with self._lock:
                    self._merge(params[0])
        elif method == "notify_klippy_disconnected":
            log.warning("Klipper disconnected from Moonraker")
            with self._lock:
                self.ws_enabled = False
        elif method == "notify_klippy_ready":
            log.info("Klipper ready")

    def _merge(self, update: dict) -> None:
        # Moonraker sends only changed fields — deep merge into the cache.
        for key, value in update.items():
            if key in self._state_cache and isinstance(self._state_cache[key], dict) and isinstance(value, dict):
                self._state_cache[key].update(value)
            else:
                self._state_cache[key] = value

    # ------------------------------------------------------------------
    # REST layer (metadata + fallback polling)
    # ------------------------------------------------------------------

    def _get(self, path: str) -> Any:
        if not _HAS_REQUESTS:
            return None
        try:
            r = _requests.get(f"{self.base_url}{path}", timeout=self._timeout)  # type: ignore[union-attr]
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            log.debug("Moonraker GET %s failed: %s", path, exc)
            return None

    def _get_metadata(self, filename: str) -> dict:
        if not filename:
            return {}
        if filename in self._metadata_cache:
            return self._metadata_cache[filename]
        raw = self._get(f"/server/files/metadata?filename={quote(filename, safe='/')}")
        if not raw:
            return {}
        result = raw.get("result", raw)
        self._metadata_cache = {filename: result}  # only keep the current file
        return result

    def _get_estimated_time(self, metadata: dict) -> int:
        return int(metadata.get("estimated_time", 0))

    def _get_preview_url(self, metadata: dict) -> str:
        thumbnails = metadata.get("thumbnails", [])
        if not isinstance(thumbnails, list) or not thumbnails:
            return ""
        # Pick the largest thumbnail to avoid a blurry upscale.
        best = max(thumbnails, key=lambda t: t.get("width", 0) * t.get("height", 0))
        thumb_path = best.get("relative_path", "")
        if thumb_path:
            return f"/server/files/gcodes/{quote(thumb_path, safe='/')}"
        return ""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return current printer state. Uses WebSocket cache when connected, REST otherwise."""
        if not self.enabled:
            return {}
        if self.ws_enabled:
            return self._state_from_cache()
        return self._state_from_rest()

    def _state_from_cache(self) -> dict:
        with self._lock:
            cache = {k: dict(v) if isinstance(v, dict) else v for k, v in self._state_cache.items()}

        ps = cache.get("print_stats", {})
        ds = cache.get("display_status", {})
        bed = cache.get("heater_bed", {})
        ext = cache.get("extruder", {})

        progress = float(ds.get("progress", 0.0) or 0.0)
        print_duration = float(ps.get("print_duration", 0.0) or 0.0)
        remaining_s = max(int((print_duration / progress) - print_duration), 0) if progress > 0 and print_duration > 0 else 0
        filename = ps.get("filename", "")
        metadata = self._get_metadata(filename)

        return {
            "state": ps.get("state", "unknown"),
            "filename": filename,
            "progress": round(progress * 100, 1),
            "elapsed_s": int(print_duration),
            "remaining_s": remaining_s,
            "estimated_s": self._get_estimated_time(metadata),
            "bed_temp": round(float(bed.get("temperature", 0.0)), 1),
            "bed_target": round(float(bed.get("target", 0.0)), 1),
            "nozzle_temp": round(float(ext.get("temperature", 0.0)), 1),
            "nozzle_target": round(float(ext.get("target", 0.0)), 1),
            "preview_url": self._get_preview_url(metadata),
        }

    def _state_from_rest(self) -> dict:
        data = self._get(
            "/printer/objects/query"
            "?print_stats&display_status&heater_bed&extruder"
        )
        if not data:
            return {}

        result = data.get("result", {}).get("status", {})
        ps = result.get("print_stats", {})
        ds = result.get("display_status", {})
        bed = result.get("heater_bed", {})
        ext = result.get("extruder", {})

        progress = float(ds.get("progress", 0.0) or 0.0)
        print_duration = float(ps.get("print_duration", 0.0) or 0.0)
        remaining_s = max(int((print_duration / progress) - print_duration), 0) if progress > 0 and print_duration > 0 else 0
        filename = ps.get("filename", "")
        metadata = self._get_metadata(filename)

        return {
            "state": ps.get("state", "unknown"),
            "filename": filename,
            "progress": round(progress * 100, 1),
            "elapsed_s": int(print_duration),
            "remaining_s": remaining_s,
            "estimated_s": self._get_estimated_time(metadata),
            "bed_temp": round(float(bed.get("temperature", 0.0)), 1),
            "bed_target": round(float(bed.get("target", 0.0)), 1),
            "nozzle_temp": round(float(ext.get("temperature", 0.0)), 1),
            "nozzle_target": round(float(ext.get("target", 0.0)), 1),
            "preview_url": self._get_preview_url(metadata),
        }
