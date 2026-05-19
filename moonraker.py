import logging
import os
from typing import Any
from urllib.parse import quote

log = logging.getLogger(__name__)

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ModuleNotFoundError:
    _HAS_REQUESTS = False
    _requests = None  # type: ignore


class MoonrakerClient:
    """Thin Moonraker REST API client for Klipper integration."""

    def __init__(self) -> None:
        # Only used for printer status and preview metadata lookup.
        self.base_url = os.getenv("MOONRAKER_URL", "").rstrip("/")
        self._timeout = 3
        self._preview_cache: dict[str, str] = {}
        self.enabled = bool(self.base_url and _HAS_REQUESTS)
        if self.enabled:
            log.info("Moonraker client enabled: %s", self.base_url)
        elif not self.base_url:
            log.info("Moonraker not configured. Set MOONRAKER_URL to enable.")

    def _get(self, path: str) -> Any:
        # Small GET wrapper with fail-soft behavior.
        try:
            r = _requests.get(  # type: ignore[union-attr]
                f"{self.base_url}{path}", timeout=self._timeout
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            log.debug("Moonraker GET %s failed: %s", path, exc)
            return None

    def _get_estimated_time(self, filename: str) -> int:
        # Get estimated print time from file metadata.
        if not filename:
            return 0
        
        metadata = self._get(
            f"/server/files/metadata?filename={quote(filename, safe='/')}"
        )
        if not metadata:
            return 0

        result = metadata.get("result", metadata)
        estimated_time = result.get("estimated_time", 0)
        return int(estimated_time)

    def _get_preview_url(self, filename: str) -> str:
        # Get preview thumbnail URL if available from Moonraker metadata.
        if not filename:
            return ""
        
        metadata = self._get(
            f"/server/files/metadata?filename={quote(filename, safe='/')}"
        )
        if not metadata:
            return ""

        result = metadata.get("result", metadata)
        # Moonraker may provide a thumbnail key with relative path
        thumbnail = result.get("thumbnails", [])
        if isinstance(thumbnail, list) and len(thumbnail) > 0:
            thumb_path = thumbnail[0].get("relative_path", "")
            if thumb_path:
                return f"/server/files/gcodes/{quote(thumb_path, safe='/')}"
        return ""

    def get_state(self) -> dict:
        """Return a dict of printer state from Moonraker, or {} if unavailable."""
        # The dashboard polls this endpoint to show read-only printer info.
        if not self.enabled:
            return {}

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
        remaining_s = 0

        if progress > 0 and print_duration > 0:
            remaining_s = max(int((print_duration / progress) - print_duration), 0)

        filename = ps.get("filename", "")

        return {
            "state": ps.get("state", "unknown"),        # printing/standby/paused/complete/error
            "filename": filename,
            "progress": round(progress * 100, 1),
            "elapsed_s": int(print_duration),
            "remaining_s": remaining_s,
            "estimated_s": self._get_estimated_time(filename),
            "bed_temp": round(bed.get("temperature", 0.0), 1),
            "bed_target": round(bed.get("target", 0.0), 1),
            "nozzle_temp": round(ext.get("temperature", 0.0), 1),
            "nozzle_target": round(ext.get("target", 0.0), 1),
            "preview_url": self._get_preview_url(filename),
        }
