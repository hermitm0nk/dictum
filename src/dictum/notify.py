"""Desktop notifications via D-Bus (org.freedesktop.Notifications)."""

from __future__ import annotations

import asyncio
import logging

log = logging.getLogger("dictum.notify")

# D-Bus constants
_NOTIF_SERVICE = "org.freedesktop.Notifications"
_NOTIF_PATH = "/org/freedesktop/Notifications"
_NOTIF_IFACE = "org.freedesktop.Notifications"


class Notifier:
    """Send desktop notifications via D-Bus directly.

    Only one notification at a time — calling notify() closes any previous
    one.  When the daemon returns to IDLE, the dismissal is instant.
    """

    def __init__(self, app_name: str = "Dictum") -> None:
        self.app_name = app_name
        self._notify_id: int | None = None

    async def _dbus_call(self, method: str, *args: object) -> str | None:
        """Call a D-Bus method on org.freedesktop.Notifications."""
        cmd = [
            "gdbus",
            "call",
            "--session",
            "--dest",
            _NOTIF_SERVICE,
            "--object-path",
            _NOTIF_PATH,
            "--method",
            f"{_NOTIF_IFACE}.{method}",
        ]
        cmd.extend(str(a) for a in args)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                log.warning("D-Bus %s failed: %s", method, stderr.decode().strip())
                return None
            return stdout.decode().strip()
        except Exception as exc:
            log.error("D-Bus %s error: %s", method, exc)
            return None

    async def notify(
        self,
        summary: str,
        body: str = "",
        icon: str = "audio-input-microphone",
        urgency: str = "normal",
        timeout: int = -1,
    ) -> int | None:
        """Send a notification (closes any previous one).

        timeout=-1 means use server default (usually 5s). timeout=0 means
        the notification stays until explicitly dismissed.
        """
        # Always close previous notification first
        await self.close()

        urgency_byte: int = {"low": 0, "normal": 1, "critical": 2}.get(urgency, 1)

        raw = await self._dbus_call(
            "Notify",
            self.app_name,  # app_name
            "uint32 0",  # replaces_id (0 = new, don't replace)
            icon,  # app_icon
            summary,  # summary
            body,  # body
            "[]",  # actions (empty)
            f"{{'urgency': <byte {urgency_byte}>}}",  # hints
            f"int32 {timeout}",  # expire_timeout
        )

        if raw and "(" in raw:
            # gdbus returns: (uint32 42,)
            try:
                id_str = raw.strip("()").split(",")[0].strip().split()[-1]
                self._notify_id = int(id_str)
                return self._notify_id
            except (ValueError, IndexError):
                pass
        return None

    async def close(self) -> None:
        """Dismiss the current notification immediately."""
        if self._notify_id is None:
            return
        await self._dbus_call(
            "CloseNotification",
            f"uint32 {self._notify_id}",
        )
        self._notify_id = None
