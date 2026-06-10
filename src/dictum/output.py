"""Output backends — wtype, wl-copy, stdout, file."""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path

from dictum.models import DictationResult, ResultTarget

log = logging.getLogger(__name__)


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
        subprocess.run(["which", name], capture_output=True, timeout=2)
        return True
    except Exception:
        return False


class OutputSink:
    """Deliver text to the requested target."""

    async def deliver(self, result: DictationResult, target: ResultTarget) -> None:
        text = result.final_text
        if not text:
            log.warning("No text to deliver")
            return

        if target == ResultTarget.PASTE:
            await self._paste(text)
        elif target == ResultTarget.CLIPBOARD:
            await self._clipboard(text)
        elif target == ResultTarget.FILE:
            await self._file(text, result.output_path)
        elif target == ResultTarget.STDOUT:
            print(text)
        # ResultTarget.NONE — do nothing

    # ---- paste: type into focused window ----

    async def _paste(self, text: str) -> None:
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
