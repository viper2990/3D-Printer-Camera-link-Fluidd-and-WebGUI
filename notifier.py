import json
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
        self.webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
        self.enabled = bool(self.webhook_url and _HAS_REQUESTS)
        if self.enabled:
            log.info("Discord notifier enabled.")
        else:
            log.info("Discord notifier disabled. Set DISCORD_WEBHOOK_URL to enable.")

    @staticmethod
    def _build_fields(facts: list | None) -> list:
        # Convert facts list to Discord embed fields.
        # First two facts display inline (side by side), rest full width.
        if not facts:
            return []
        return [
            {"name": f["title"], "value": str(f["value"]), "inline": i < 2}
            for i, f in enumerate(facts)
        ]

    def _build_embed(self, title: str, message: str, color: int,
                     emoji: str, facts: list | None) -> dict:
        embed: dict = {
            "title": f"{emoji} {title}".strip() if emoji else title,
            "description": message,
            "color": color,
        }
        fields = self._build_fields(facts)
        if fields:
            embed["fields"] = fields
        return embed

    def send(self, title: str, message: str, color: int = 0xFF4444,
             emoji: str = ":rotating_light:", facts: list | None = None) -> "str | None":
        """Send a notification. Returns the Discord message ID on success, None on failure."""
        if not self.enabled:
            return None
        try:
            embed = self._build_embed(title, message, color, emoji, facts)
            r = _requests.post(  # type: ignore[union-attr]
                self.webhook_url + "?wait=true",
                json={"embeds": [embed]},
                timeout=5,
            )
            r.raise_for_status()
            log.info("Discord notification sent: %s", title)
            return r.json().get("id")
        except Exception as exc:
            log.warning("Discord notification failed: %s", exc)
            return None

    def send_with_image(self, title: str, message: str, image_bytes: bytes,
                        color: int = 0xFF4444, emoji: str = ":rotating_light:",
                        facts: list | None = None) -> "str | None":
        """Send a notification with image. Returns the Discord message ID on success, None on failure."""
        if not self.enabled:
            return None
        try:
            embed = self._build_embed(title, message, color, emoji, facts)
            embed["image"] = {"url": "attachment://snapshot.jpg"}
            r = _requests.post(  # type: ignore[union-attr]
                self.webhook_url + "?wait=true",
                data={"payload_json": json.dumps({"embeds": [embed]})},
                files={"file": ("snapshot.jpg", image_bytes, "image/jpeg")},
                timeout=10,
            )
            r.raise_for_status()
            log.info("Discord notification with image sent: %s", title)
            return r.json().get("id")
        except Exception as exc:
            log.warning("Discord notification with image failed: %s", exc)
            return None

    def edit_message(self, message_id: str, title: str, message: str,
                     color: int = 0x4db7ff, emoji: str = "",
                     facts: list | None = None) -> bool:
        """Edit an existing Discord webhook message in place (live countdown updates)."""
        if not self.enabled or not message_id:
            return False
        try:
            # Derive edit URL from webhook URL: .../webhooks/{id}/{token}/messages/{msg_id}
            edit_url = self.webhook_url.rstrip("/") + f"/messages/{message_id}"
            embed = self._build_embed(title, message, color, emoji, facts)
            r = _requests.patch(  # type: ignore[union-attr]
                edit_url,
                json={"embeds": [embed]},
                timeout=5,
            )
            r.raise_for_status()
            return True
        except Exception as exc:
            log.debug("Discord message edit failed: %s", exc)
            return False


class TeamsNotifier:
    """
    Push notifications to Microsoft Teams via incoming webhook.

    How to set up (new Teams / Microsoft 365):
      1. Open the Teams channel you want notifications in.
      2. Click the + icon next to the channel name → search for "Workflows".
      3. Choose "Post to a channel when a webhook request is received".
      4. Follow the prompts — it creates a Power Automate flow.
      5. Copy the webhook URL shown at the end.
      6. Paste it as TEAMS_WEBHOOK_URL in start_monitor.sh.

    Note: Teams webhooks do not support binary file uploads, so camera
    snapshots are not included. You will get the same text alerts as Discord
    but without the attached image.
    """

    def __init__(self) -> None:
        self.webhook_url = os.getenv("TEAMS_WEBHOOK_URL", "").strip()
        self.enabled = bool(self.webhook_url and _HAS_REQUESTS)
        if self.enabled:
            log.info("Teams notifier enabled.")
        else:
            log.info("Teams notifier disabled. Set TEAMS_WEBHOOK_URL to enable.")

    def send(self, title: str, message: str, emoji: str = "",
             facts: list | None = None) -> bool:
        """Send a Teams notification using Adaptive Card format.

        facts: optional list of {"title": str, "value": str} shown as a
               key/value table below the message — useful for time remaining,
               progress, filename, etc.
        """
        if not self.enabled:
            return False
        try:
            full_title = f"{emoji} {title}".strip() if emoji else title

            # Split on newlines so each line becomes its own TextBlock
            # (Adaptive Cards ignore \\n inside a single TextBlock).
            body: list = [
                {
                    "type": "TextBlock",
                    "size": "Medium",
                    "weight": "Bolder",
                    "text": full_title,
                    "wrap": True,
                }
            ]
            for line in message.split("\n"):
                line = line.strip()
                if line:
                    body.append({"type": "TextBlock", "text": line, "wrap": True})

            if facts:
                body.append({
                    "type": "FactSet",
                    "facts": [{"title": f["title"], "value": str(f["value"])} for f in facts],
                })

            payload = {
                "type": "message",
                "attachments": [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "contentUrl": None,
                        "content": {
                            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                            "type": "AdaptiveCard",
                            "version": "1.2",
                            "body": body,
                        },
                    }
                ],
            }
            r = _requests.post(  # type: ignore[union-attr]
                self.webhook_url,
                json=payload,
                timeout=5,
            )
            r.raise_for_status()
            log.info("Teams notification sent: %s", title)
            return True
        except Exception as exc:
            log.warning("Teams notification failed: %s", exc)
            return False
