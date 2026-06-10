"""Dictum CLI — thin IPC client for Hyprland/Sway hotkeys."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine
from typing import Annotated, Any, TypeVar

import typer
from rich.console import Console

from dictum.ipc import IpcError, send_request

app = typer.Typer(no_args_is_help=True, help="Wayland hotkey dictation CLI.")
console = Console(stderr=True)

T = TypeVar("T")


def _run(coro: Coroutine[Any, Any, T]) -> T:
    """Shim for running async from sync CLI commands."""
    return asyncio.run(coro)


def _die(msg: str, code: int = 1) -> None:
    console.print(f"[red]{msg}[/red]")
    raise typer.Exit(code)


def _ipc(cmd: str, **kw: object) -> dict[str, Any]:
    """Send IPC request, die on error."""
    try:
        return _run(send_request(cmd, **kw))
    except IpcError as exc:
        _die(str(exc))
        return {}  # unreachable but satisfies type checker


# ──────────────────────────────────────────────
# daemon
# ──────────────────────────────────────────────


@app.command()
def daemon(
    profile: Annotated[str, typer.Option(help="Default profile name.")] = "default",
) -> None:
    """Run the daemon in the foreground."""
    from dictum.daemon import run_daemon

    run_daemon(profile_name=profile)


# ──────────────────────────────────────────────
# status
# ──────────────────────────────────────────────


@app.command()
def status(
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Print daemon state."""
    data = _ipc("status")
    if json_out:
        console.print_json(data=data)
    else:
        console.print(f"state: {data.get('state', '?')}")
        console.print(f"profile: {data.get('active_profile', '?')}")
        if data.get("last_error"):
            console.print(f"error: {data['last_error']}")


# ──────────────────────────────────────────────
# toggle
# ──────────────────────────────────────────────


@app.command()
def toggle(
    profile: Annotated[str, typer.Option(help="Profile name.")] = "default",
    result: Annotated[
        str, typer.Option("--result", help="Output target: paste/clipboard/stdout/file/none")
    ] = "paste",
) -> None:
    """Toggle recording via the daemon.

    First press starts recording; second press stops and processes.
    """
    data = _ipc("toggle", profile=profile, target=result)

    if not data.get("ok"):
        _die(data.get("error", "Toggle failed"))

    if data.get("started"):
        console.print("[green]● Recording…[/green] (press again to stop)")
    elif data.get("stopped_recording"):
        console.print("[blue]● Processing…[/blue]")


# ──────────────────────────────────────────────
# start / stop / cancel
# ──────────────────────────────────────────────


@app.command()
def start(
    profile: Annotated[str, typer.Option(help="Profile name.")] = "default",
) -> None:
    """Start recording through the daemon."""
    data = _ipc("start", profile=profile)
    if not data.get("ok"):
        _die(data.get("error", "Start failed"))
    console.print("[green]● Recording…[/green]")


@app.command()
def stop(
    result: Annotated[str, typer.Option("--result", help="Output target.")] = "paste",
) -> None:
    """Stop recording and process."""
    data = _ipc("stop", target=result)
    if not data.get("ok"):
        _die(data.get("error", "Stop failed"))
    console.print("[blue]● Processing…[/blue]")


@app.command()
def cancel() -> None:
    """Cancel current operation and return to idle."""
    _ipc("cancel")
    console.print("[dim]Cancelled.[/dim]")


# ──────────────────────────────────────────────
# once (one-shot, no daemon needed)
# ──────────────────────────────────────────────


@app.command()
def once(
    profile: Annotated[str, typer.Option(help="Profile name.")] = "default",
    result: Annotated[str, typer.Option("--result", help="Output target.")] = "paste",
    duration: Annotated[int, typer.Option("--duration", "-d", help="Max recording seconds.")] = 30,
) -> None:
    """Run one dictation job in the current process (no daemon needed).

    Records for up to DURATION seconds or until silence, transcribes,
    polishes, and delivers the result.
    """
    from dictum.asr import ParakeetASR
    from dictum.llm import OpenAILLM
    from dictum.models import DictationResult, Profile, ResultTarget
    from dictum.output import OutputSink
    from dictum.recorder import Recorder

    console.print("[dim]Recording… (speak, then wait for silence)[/dim]")

    rec = Recorder(silence_timeout=1.5)
    rec.start()

    deadline = time.monotonic() + duration
    while rec.is_recording and time.monotonic() < deadline:
        time.sleep(0.1)

    if rec.is_recording:
        console.print("[dim]Max duration reached, stopping…[/dim]")

    audio_path = rec.stop()
    console.print(f"[dim]Saved {audio_path.stat().st_size} bytes[/dim]")

    # Transcribe
    console.print("[dim]Transcribing…[/dim]")
    asr = ParakeetASR()
    transcript = _run(asr.transcribe(audio_path))
    console.print(f"[green]Transcript:[/green] {transcript.text}")

    # Polish
    prof = Profile(name=profile)
    polished: str | None = None
    if prof.llm:
        llm = OpenAILLM()
        try:
            console.print("[dim]Polishing…[/dim]")
            polished = _run(llm.polish(transcript, prof))
            console.print(f"[green]Polished:[/green] {polished}")
        except Exception as exc:
            console.print(f"[yellow]LLM failed ({exc}), using raw transcript[/yellow]")

    # Output
    target = ResultTarget(result)
    dr = DictationResult(transcript=transcript, polished_text=polished, target=target)
    _run(OutputSink().deliver(dr, target))

    if target == ResultTarget.STDOUT:
        print(dr.final_text)


if __name__ == "__main__":
    app()
