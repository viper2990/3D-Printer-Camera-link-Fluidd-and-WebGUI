import logging
import math
import os
import time
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


class PrintAnomalyDetector:
    """
    Heuristic camera-based detector for spaghetti and blob print failures.

    Spaghetti: detects a sudden spike in edge density relative to recent checks.
               Uses a rolling window so the detector naturally tracks print growth
               and only fires on unexpected jumps — not gradual layer accumulation.

    Blob: large compact mass of oozing filament detected via a connected region of
          significant pixel change against a slowly-updating reference frame.

    No ML or external service required — runs entirely on-device using OpenCV.
    Tune sensitivity with env vars if you get false positives.
    """

    def __init__(self) -> None:
        self.enabled = os.getenv("DETECTOR_ENABLED", "true").lower() == "true"
        self.check_interval_s = max(float(os.getenv("DETECTOR_INTERVAL_S", "30")), 5.0)
        # How many checks to collect before detection begins (fills the rolling window).
        self.warmup_checks = max(int(os.getenv("DETECTOR_WARMUP", "5")), 2)
        self.consecutive_required = max(int(os.getenv("DETECTOR_CONSECUTIVE", "2")), 1)
        self.cooldown_s = max(float(os.getenv("DETECTOR_COOLDOWN_S", "300")), 60.0)
        # Spaghetti fires when edge density exceeds (rolling average × this multiplier).
        # Raise if tall prints still trigger false positives.
        self._spaghetti_multiplier = max(float(os.getenv("DETECTOR_SPAGHETTI_MULTIPLIER", "2.5")), 1.1)
        # Fraction of ROI area a blob contour must cover.
        self._blob_area_min = min(max(float(os.getenv("DETECTOR_BLOB_AREA_MIN", "0.12")), 0.01), 0.5)
        # Where the detection ROI starts (fraction from top of frame).
        # 0.65 = look only at the bottom 35% of the frame, below the growing print.
        self._roi_start = min(max(float(os.getenv("DETECTOR_ROI_START", "0.65")), 0.0), 0.9)
        # How many recent edge density samples form the rolling baseline.
        self._window_size = max(int(os.getenv("DETECTOR_WINDOW_SIZE", "8")), 3)

        self._reset_state()

    def _reset_state(self) -> None:
        self._last_check = 0.0
        self._last_spaghetti_alert = 0.0
        self._last_blob_alert = 0.0
        self._spaghetti_hits = 0
        self._blob_hits = 0
        self._check_count = 0
        self._edge_window: list[float] = []   # rolling edge density history
        self._reference_frame: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._reset_state()
        log.info("Anomaly detector reset for new print")

    def should_check(self) -> bool:
        return self.enabled and (time.monotonic() - self._last_check) >= self.check_interval_s

    def analyze(self, jpeg_bytes: bytes) -> tuple[bool, bool]:
        """
        Analyze a JPEG frame.
        Returns (spaghetti_alert, blob_alert) — True means the alert should fire now.
        """
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return False, False
        self._last_check = time.monotonic()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        h, w = blurred.shape

        roi_top = int(h * self._roi_start)
        roi = blurred[roi_top:, :]

        # --- Spaghetti: Canny edge density vs rolling window average ---
        edges = cv2.Canny(roi, 30, 90)
        edge_density = float(np.count_nonzero(edges)) / edges.size

        # --- Blob: large compact region vs slowly-updated reference frame ---
        blob_raw = False
        if self._reference_frame is not None:
            diff = cv2.absdiff(roi, self._reference_frame)
            _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            roi_area = roi.shape[0] * roi.shape[1]
            for c in contours:
                area = cv2.contourArea(c)
                if area < roi_area * self._blob_area_min:
                    continue
                perimeter = cv2.arcLength(c, True)
                if perimeter == 0:
                    continue
                compactness = (4 * math.pi * area) / (perimeter ** 2)
                if compactness > 0.40:
                    blob_raw = True
                    break

        self._check_count += 1

        # Warmup: fill the rolling window and seed the reference frame.
        if self._check_count <= self.warmup_checks:
            self._edge_window.append(edge_density)
            self._reference_frame = roi.copy()
            if self._check_count == self.warmup_checks:
                log.info("Detector ready: avg_edge_density=%.4f (window=%d)",
                         sum(self._edge_window) / len(self._edge_window), self._window_size)
            return False, False

        # Spaghetti: compare current density to rolling average of recent checks.
        # The window excludes the current sample so we compare against history only.
        if len(self._edge_window) >= 2:
            rolling_avg = sum(self._edge_window) / len(self._edge_window)
            spaghetti_raw = edge_density > max(rolling_avg, 0.001) * self._spaghetti_multiplier
        else:
            spaghetti_raw = False

        # Update rolling window (keep at most _window_size entries).
        self._edge_window.append(edge_density)
        if len(self._edge_window) > self._window_size:
            self._edge_window.pop(0)

        # Adapt reference frame to normal print growth.
        if self._reference_frame is not None:
            self._reference_frame = cv2.addWeighted(self._reference_frame, 0.85, roi, 0.15, 0)

        if spaghetti_raw:
            self._spaghetti_hits = min(self._spaghetti_hits + 1, self.consecutive_required + 1)
        else:
            self._spaghetti_hits = max(self._spaghetti_hits - 1, 0)

        if blob_raw:
            self._blob_hits = min(self._blob_hits + 1, self.consecutive_required + 1)
        else:
            self._blob_hits = max(self._blob_hits - 1, 0)

        now = time.monotonic()
        spaghetti_alert = (
            self._spaghetti_hits >= self.consecutive_required
            and (now - self._last_spaghetti_alert) >= self.cooldown_s
        )
        blob_alert = (
            self._blob_hits >= self.consecutive_required
            and (now - self._last_blob_alert) >= self.cooldown_s
        )

        if spaghetti_alert:
            self._last_spaghetti_alert = now
            self._spaghetti_hits = 0
            rolling_avg = sum(self._edge_window) / len(self._edge_window) if self._edge_window else 0
            log.warning("Spaghetti detected (edge_density=%.4f, rolling_avg=%.4f, multiplier=%.1f)",
                        edge_density, rolling_avg, self._spaghetti_multiplier)

        if blob_alert:
            self._last_blob_alert = now
            self._blob_hits = 0
            log.warning("Blob detected (compact contour exceeded %.0f%% of ROI area)", self._blob_area_min * 100)

        return spaghetti_alert, blob_alert
