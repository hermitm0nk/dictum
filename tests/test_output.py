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
    calls: list[tuple[list[str], bytes | None]] = []

    async def fake_run(cmd: list[str], input_data: bytes | None = None) -> str:
        calls.append((cmd, input_data))
        return ""

    monkeypatch.setattr(output_mod, "_run", fake_run)
    monkeypatch.setattr(output_mod, "_has_binary", lambda name: True)
    monkeypatch.setattr(OutputSink, "_read_clipboard", lambda self: _immediate(None))
    monkeypatch.setattr(output_mod.asyncio, "sleep", _no_sleep())
    profile = Profile()
    profile.output.paste_method = PasteMethod.CLIPBOARD
    _run(OutputSink().deliver(_result(), ResultTarget.PASTE, profile))
    assert calls[0] == (["wl-copy"], b"hello")
    assert calls[1][0] == ["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"]


def test_auto_routes_xwayland_to_clipboard(monkeypatch) -> None:
    calls: list[list[str]] = []

    async def fake_run(cmd: list[str], input_data: bytes | None = None) -> str:
        calls.append(cmd)
        return ""

    async def fake_xwayland() -> bool:
        return True

    monkeypatch.setattr(output_mod, "_run", fake_run)
    monkeypatch.setattr(output_mod, "_has_binary", lambda name: True)
    monkeypatch.setattr(output_mod, "focused_window_is_xwayland", fake_xwayland)
    monkeypatch.setattr(OutputSink, "_read_clipboard", lambda self: _immediate(None))
    monkeypatch.setattr(output_mod.asyncio, "sleep", _no_sleep())
    _run(OutputSink().deliver(_result(), ResultTarget.PASTE, Profile()))
    assert ["wl-copy"] in calls
    assert ["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"] in calls


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
    copied: list[bytes | None] = []

    async def fake_run(cmd: list[str], input_data: bytes | None = None) -> str:
        if cmd == ["wl-copy"]:
            copied.append(input_data)
        return ""

    monkeypatch.setattr(output_mod, "_run", fake_run)
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


def _no_sleep():
    async def _sleep(_delay: float) -> None:
        return None

    return _sleep
