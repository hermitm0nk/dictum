"""Paste routing: typing vs clipboard, with XWayland auto-detection."""

import asyncio

from dictum import output as output_mod
from dictum.models import DictationResult, PasteMethod, Profile, ResultTarget, Transcript
from dictum.output import OutputSink


def _result(text: str = "hello") -> DictationResult:
    return DictationResult(transcript=Transcript(text=text), target=ResultTarget.PASTE)


def _run(coro):
    return asyncio.run(coro)


def test_default_paste_method_is_auto() -> None:
    assert Profile().output.paste_method == PasteMethod.AUTO


def test_type_method_uses_wtype(monkeypatch) -> None:
    calls: list[list[str]] = []

    async def fake_run(cmd: list[str], input_data: bytes | None = None) -> str:
        calls.append(cmd)
        return ""

    monkeypatch.setattr(output_mod, "_run", fake_run)
    monkeypatch.setattr(output_mod, "_has_binary", lambda name: True)
    profile = Profile()
    profile.output.paste_method = PasteMethod.TYPE
    _run(OutputSink().deliver(_result(), ResultTarget.PASTE, profile))
    assert calls == [["wtype", "--", "hello"]]


def test_clipboard_method_copies_and_sends_ctrl_v(monkeypatch) -> None:
    runs: list[list[str]] = []
    feeds: list[tuple[list[str], bytes]] = []

    async def fake_run(cmd: list[str], input_data: bytes | None = None) -> str:
        runs.append(cmd)
        return ""

    async def fake_feed(cmd: list[str], input_data: bytes) -> None:
        feeds.append((cmd, input_data))

    monkeypatch.setattr(output_mod, "_run", fake_run)
    monkeypatch.setattr(output_mod, "_run_feed", fake_feed)
    monkeypatch.setattr(output_mod, "_has_binary", lambda name: True)
    monkeypatch.setattr(OutputSink, "_read_clipboard", lambda self: _immediate(None))
    monkeypatch.setattr(output_mod.asyncio, "sleep", _no_sleep())
    profile = Profile()
    profile.output.paste_method = PasteMethod.CLIPBOARD
    _run(OutputSink().deliver(_result(), ResultTarget.PASTE, profile))
    assert feeds == [(["wl-copy"], b"hello")]
    assert runs == [["ydotool", "key", "29:1", "47:1", "47:0", "29:0"]]


def test_clipboard_paste_key_falls_back_to_wtype(monkeypatch) -> None:
    runs: list[list[str]] = []

    async def fake_run(cmd: list[str], input_data: bytes | None = None) -> str:
        runs.append(cmd)
        return ""

    async def fake_feed(cmd: list[str], input_data: bytes) -> None:
        return None

    # ydotool missing, wtype present -> wtype sends Ctrl+V.
    monkeypatch.setattr(output_mod, "_run", fake_run)
    monkeypatch.setattr(output_mod, "_run_feed", fake_feed)
    monkeypatch.setattr(output_mod, "_has_binary", lambda name: name != "ydotool")
    monkeypatch.setattr(OutputSink, "_read_clipboard", lambda self: _immediate(None))
    monkeypatch.setattr(output_mod.asyncio, "sleep", _no_sleep())
    _run(OutputSink()._clipboard_paste("hello"))
    assert runs == [["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"]]


def test_auto_routes_xwayland_to_clipboard(monkeypatch) -> None:
    runs: list[list[str]] = []
    feeds: list[tuple[list[str], bytes]] = []

    async def fake_run(cmd: list[str], input_data: bytes | None = None) -> str:
        runs.append(cmd)
        return ""

    async def fake_feed(cmd: list[str], input_data: bytes) -> None:
        feeds.append((cmd, input_data))

    async def fake_xwayland() -> bool:
        return True

    monkeypatch.setattr(output_mod, "_run", fake_run)
    monkeypatch.setattr(output_mod, "_run_feed", fake_feed)
    monkeypatch.setattr(output_mod, "_has_binary", lambda name: True)
    monkeypatch.setattr(output_mod, "focused_window_is_xwayland", fake_xwayland)
    monkeypatch.setattr(OutputSink, "_read_clipboard", lambda self: _immediate(None))
    monkeypatch.setattr(output_mod.asyncio, "sleep", _no_sleep())
    _run(OutputSink().deliver(_result(), ResultTarget.PASTE, Profile()))
    assert feeds == [(["wl-copy"], b"hello")]
    assert ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"] in runs


def test_auto_routes_native_wayland_to_typing(monkeypatch) -> None:
    calls: list[list[str]] = []

    async def fake_run(cmd: list[str], input_data: bytes | None = None) -> str:
        calls.append(cmd)
        return ""

    async def fake_native() -> bool:
        return False

    monkeypatch.setattr(output_mod, "_run", fake_run)
    monkeypatch.setattr(output_mod, "_has_binary", lambda name: True)
    monkeypatch.setattr(output_mod, "focused_window_is_xwayland", fake_native)
    _run(OutputSink().deliver(_result(), ResultTarget.PASTE, Profile()))
    assert calls == [["wtype", "--", "hello"]]


def test_clipboard_paste_restores_previous_clipboard(monkeypatch) -> None:
    copied: list[bytes] = []

    async def fake_feed(cmd: list[str], input_data: bytes) -> None:
        if cmd == ["wl-copy"]:
            copied.append(input_data)

    async def fake_run(cmd: list[str], input_data: bytes | None = None) -> str:
        return ""

    monkeypatch.setattr(output_mod, "_run", fake_run)
    monkeypatch.setattr(output_mod, "_run_feed", fake_feed)
    monkeypatch.setattr(output_mod, "_has_binary", lambda name: True)
    monkeypatch.setattr(OutputSink, "_read_clipboard", lambda self: _immediate("old"))
    monkeypatch.setattr(output_mod.asyncio, "sleep", _no_sleep())
    _run(OutputSink()._clipboard_paste("new"))
    assert copied == [b"new", b"old"]


def test_clipboard_paste_falls_back_to_typing_without_binaries(monkeypatch) -> None:
    calls: list[list[str]] = []

    async def fake_run(cmd: list[str], input_data: bytes | None = None) -> str:
        calls.append(cmd)
        return ""

    monkeypatch.setattr(output_mod, "_run", fake_run)
    # wl-copy missing -> fall back to typing; wtype present for the fallback.
    monkeypatch.setattr(output_mod, "_has_binary", lambda name: name == "wtype")
    _run(OutputSink()._clipboard_paste("hello"))
    assert calls == [["wtype", "--", "hello"]]


def _immediate(value):
    async def _get(self=None):
        return value

    return _get()


def test_run_feed_detaches_outputs(monkeypatch) -> None:
    """wl-copy must not be awaited on piped outputs: it daemonizes and
    holds inherited fds open, hanging communicate() until timeout."""
    seen: dict = {}

    class FakeProc:
        returncode = 0

        async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
            seen["input"] = input
            return (b"", b"")

    async def fake_exec(*cmd: str, **kwargs):
        seen.update(kwargs)
        seen["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(output_mod.asyncio, "create_subprocess_exec", fake_exec)
    _run(output_mod._run_feed(["wl-copy"], b"hi"))
    assert seen["cmd"] == ["wl-copy"]
    assert seen["input"] == b"hi"
    assert seen["stdout"] == output_mod.asyncio.subprocess.DEVNULL
    assert seen["stderr"] == output_mod.asyncio.subprocess.DEVNULL


def test_run_feed_reports_timeout(monkeypatch) -> None:
    class HangingProc:
        returncode = None
        killed = False

        async def communicate(self, input: bytes | None = None):
            raise AssertionError("should be cancelled by wait_for")

        def kill(self) -> None:
            self.killed = True

    proc = HangingProc()

    async def fake_exec(*cmd: str, **kwargs):
        return proc

    async def fake_wait_for(awaitable, timeout: float):
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise output_mod.asyncio.TimeoutError()

    monkeypatch.setattr(output_mod.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(output_mod.asyncio, "wait_for", fake_wait_for)
    try:
        _run(output_mod._run_feed(["wl-copy"], b"hi"))
    except RuntimeError as exc:
        assert "timed out" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
    assert proc.killed


def _no_sleep():
    async def _sleep(_delay: float) -> None:
        return None

    return _sleep
