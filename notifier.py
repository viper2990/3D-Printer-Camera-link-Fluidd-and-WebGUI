import logging
import os

log = logging.getLogger(__name__)

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ModuleNotFoundError:
    _HAS_REQUESTS = False
    _requests = None  # type: ignore


class Notifier:
    """
    Push notifications via a Discord webhook.
    Create a free webhook in any Discord channel:
    Channel Settings -> Integrations -> Webhooks -> New Webhook -> Copy URL
    Then set DISCORD_WEBHOOK_URL in start_monitor.sh.
    """

    def __init__(self) -> None:
        # Webhook is the only Discord integration used by this project.
        self.webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
        self.enabled = bool(self.webhook_url and _HAS_REQUESTS)
        if self.enabled:
            log.info("Discord notifier enabled.")
        else:
            log.info("Discord notifier disabled. Set DISCORD_WEBHOOK_URL to enable.")

    def send(self, title: str, message: str, color: int = 0xFF4444, emoji: str = ":rotating_light:") -> bool:
        # Plain embed alert, used for camera fault messages and test pings.
        if not self.enabled:
            return False
        try:
            payload = {
                "embeds": [
                    {
                        "title": f"{emoji} {title}",
                        "description": message,
                        "color": color,
                    }
                ]
            }
            r = _requests.post(  # type: ignore[union-attr]
                self.webhook_url,
                json=payload,
                timeout=5,
            )
            r.raise_for_status()
            log.info("Discord notification sent: %s", title)
            return True
        except Exception as exc:
            log.warning("Discord notification failed: %s", exc)
            return False

    def send_with_image(self, title: str, message: str, image_bytes: bytes, color: int = 0xFF4444, emoji: str = ":rotating_light:") -> bool:
        """Send a Discord notification with a camera snapshot attached."""
        # Same alert flow as send(), but includes a JPEG snapshot attachment.
        if not self.enabled:
            return False
        try:
            import json
            payload = {
                "embeds": [
                    {
                        "title": f"{emoji} {title}",
                        "description": message,
                        "color": color,
                        "image": {"url": "attachment://snapshot.jpg"},
                    }
                ]
            }
            r = _requests.post(  # type: ignore[union-attr]
                self.webhook_url,
                data={"payload_json": json.dumps(payload)},
                files={"file": ("snapshot.jpg", image_bytes, "image/jpeg")},
                timeout=10,
            )
            r.raise_for_status()
            log.info("Discord notification with image sent: %s", title)
            return True
        except Exception as exc:
            log.warning("Discord notification with image failed: %s", exc)
            return False
