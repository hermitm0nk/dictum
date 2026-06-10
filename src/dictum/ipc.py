"""IPC protocol for daemon communication over Unix domain socket."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1


# Socket lives in $XDG_RUNTIME_DIR/dictum/ when available, else ~/.cache/dictum/
def _runtime_dir() -> Path:
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg) / "dictum"
    return Path.home() / ".cache" / "dictum"


SOCKET_DIR = _runtime_dir()
SOCKET_PATH = SOCKET_DIR / "dictum.sock"


def ensure_socket_dir() -> None:
    SOCKET_DIR.mkdir(parents=True, exist_ok=True)


class IpcError(Exception):
    pass


async def send_request(command: str, **payload: Any) -> dict[str, Any]:
    """Send a JSON request to the daemon and return the parsed response."""
    ensure_socket_dir()
    if not SOCKET_PATH.exists():
        raise IpcError("Daemon is not running (socket not found)")

    msg = json.dumps({"v": PROTOCOL_VERSION, "cmd": command, **payload})

    reader, writer = await asyncio.open_unix_connection(str(SOCKET_PATH))
    try:
        writer.write((msg + "\n").encode())
        await writer.drain()

        line = await asyncio.wait_for(reader.readline(), timeout=30)
        if not line:
            raise IpcError("Daemon closed connection without response")
        return json.loads(line)  # type: ignore[no-any-return]
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


def json_response(data: dict[str, Any]) -> bytes:
    """Serialize a response dict to bytes for sending over the socket."""
    return (json.dumps({"v": PROTOCOL_VERSION, **data}) + "\n").encode()
