"""Output backends — wtype, ydotool, wl-copy, stdout, file."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path

from dictum.models import DictationResult, PasteMethod, Profile, ResultTarget

log = logging.getLogger(__name__)

# Delay after wl-copy so the compositor (and the XWayland clipboard proxy)
# picks up the new content before the synthetic Ctrl+V is sent.
CLIPBOARD_SETTLE_SECONDS = 0.15
# Delay before restoring the previous clipboard so the target app has
# time to read the pasted content after the keypress.
CLIPBOARD_RESTORE_SECONDS = 0.5


async def _run(cmd: list[str], input_data: bytes | None = None) -> str:
    """Run a command and return stdout. Raises on non-zero exit."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE if input_data else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(input=input_data), timeout=10)
    if proc.returncode != 0:
        err = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"{' '.join(cmd)} failed: {err}")
    return stdout.decode(errors="replace").strip()


def _has_binary(name: str) -> bool:
    try:
        return shutil.which(name) is not None
    except Exception:
        return False


async def focused_window_is_xwayland() -> bool:
    """True if the focused window is an XWayland (X11) client.

    Returns False when the compositor cannot be queried (no hyprctl,
    parse error, timeout) — callers then fall back to keystroke typing,
    preserving the previous behavior.
    """
    if not _has_binary("hyprctl"):
        return False
    try:
        proc = await asyncio.create_subprocess_exec(
            "hyprctl",
            "activewindow",
            "-j",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2)
        if proc.returncode != 0:
            return False
        data = json.loads(stdout.decode(errors="replace"))
        return bool(data.get("xwayland"))
    except Exception:
        return False


class OutputSink:
    """Deliver text to the requested target."""

    async def deliver(
        self,
        result: DictationResult,
        target: ResultTarget,
        profile: Profile | None = None,
    ) -> None:
        text = result.final_text
        if not text:
            log.warning("No text to deliver")
            return

        if target == ResultTarget.PASTE:
            method = (profile or Profile()).output.paste_method
            await self._paste(text, method)
        elif target == ResultTarget.CLIPBOARD:
            await self._clipboard(text)
        elif target == ResultTarget.FILE:
            await self._file(text, result.output_path)
        elif target == ResultTarget.STDOUT:
            print(text)
        # ResultTarget.NONE — do nothing

    # ---- paste: type or clipboard+Ctrl+V into focused window ----

    async def _paste(self, text: str, method: PasteMethod = PasteMethod.AUTO) -> None:
        """Deliver text to the focused window.

        Keystroke typing (wtype/ydotool) is exact on native Wayland but
        mistranslates through XWayland's XKB mapping — trigger characters
        such as `/` or `'` fire app shortcuts (e.g. Firefox quick-find)
        instead of inserting text. For XWayland windows, copying to the
        clipboard and sending a single Ctrl+V avoids per-character mapping.
        """
        if method == PasteMethod.CLIPBOARD:
            await self._clipboard_paste(text)
            return
        if method == PasteMethod.AUTO and await focused_window_is_xwayland():
            log.info("XWayland window focused, pasting via clipboard")
            await self._clipboard_paste(text)
            return
        await self._type(text)

    async def _type(self, text: str) -> None:
        """Type text into the currently focused window via wtype or ydotool."""
        if _has_binary("wtype"):
            log.info("Pasting via wtype")
            await _run(["wtype", "--", text])
        elif _has_binary("ydotool"):
            log.info("Pasting via ydotool type")
            await _run(["ydotool", "type", "--delay", "0", "--", text])
        else:
            log.warning("No wtype or ydotool found, falling back to wl-copy + stderr")
            await self._clipboard(text)

    async def _clipboard_paste(self, text: str) -> None:
        """Copy text and send Ctrl+V so the app pastes it as one unit."""
        if not (_has_binary("wl-copy") and _has_binary("wtype")):
            log.warning("wl-copy+wtype needed for clipboard paste, falling back to typing")
            await self._type(text)
            return
        saved = await self._read_clipboard()
        await _run(["wl-copy"], input_data=text.encode())
        await asyncio.sleep(CLIPBOARD_SETTLE_SECONDS)
        try:
            log.info("Pasting via clipboard + Ctrl+V")
            await _run(["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"])
        finally:
            if saved is not None:
                await asyncio.sleep(CLIPBOARD_RESTORE_SECONDS)
                try:
                    await _run(["wl-copy"], input_data=saved.encode())
                except Exception as exc:
                    log.warning("Could not restore clipboard: %s", exc)
            # saved is None (unreadable clipboard): leave the pasted text
            # in place so the user can paste again.

    async def _read_clipboard(self) -> str | None:
        """Best-effort read of the current clipboard. None when unavailable."""
        if not _has_binary("wl-paste"):
            return None
        try:
            proc = await asyncio.create_subprocess_exec(
                "wl-paste",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            if proc.returncode != 0:
                return None
            return stdout.decode(errors="replace")
        except Exception:
            return None

    # ---- file output ----

    async def _file(self, text: str, path: Path | None = None) -> None:
        out = path or Path("/tmp/dictum-output.txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        log.info("Wrote output to %s", out)

    # ---- clipboard ----

    async def _clipboard(self, text: str) -> None:
        log.info("Copying to clipboard via wl-copy")
        await _run(["wl-copy"], input_data=text.encode())
