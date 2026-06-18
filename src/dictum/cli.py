"""Dictum CLI — thin IPC client for Hyprland/Sway hotkeys."""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from collections.abc import Coroutine
from pathlib import Path
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
# native (download/manage pre-built C++ binaries)
# ──────────────────────────────────────────────

native_app = typer.Typer(
    no_args_is_help=True,
    help="Download and inspect pre-built llama.cpp / CrispASR binaries.",
)
app.add_typer(native_app, name="native")


@native_app.command("install")
def native_install(
    lib: Annotated[
        str,
        typer.Option("--lib", help="Which library: llama, crispasr, or all (default)."),
    ] = "all",
    variant: Annotated[
        str,
        typer.Option("--variant", help="Build variant: vulkan (default) or cpu."),
    ] = "vulkan",
    force: Annotated[
        bool, typer.Option("--force", help="Re-download even if already installed.")
    ] = False,
) -> None:
    """Download and extract pinned llama.cpp and/or CrispASR releases."""
    from dictum.native_installer import (
        NativeLib,
        NativeVariant,
        install_lib,
        is_installed,
    )

    try:
        v = NativeVariant(variant)
    except ValueError:
        _die(f"Invalid --variant {variant!r}; choose 'vulkan' or 'cpu'.")
        return

    if lib == "all":
        libs = [NativeLib.LLAMA, NativeLib.CRISPASR]
    elif lib in ("llama", "crispasr"):
        libs = [NativeLib(lib)]
    else:
        _die(f"Invalid --lib {lib!r}; choose 'llama', 'crispasr', or 'all'.")
        return

    failed = False
    for name in libs:
        if not force and is_installed(name, v):
            console.print(f"[green]✓[/green] {name.value} {v.value} already installed")
            continue
        try:
            path = install_lib(name, v, force=force)
            console.print(f"[green]✓[/green] Installed {name.value} → {path}")
        except Exception as exc:
            console.print(f"[red]✗[/red] {name.value}: {exc}")
            failed = True

    if failed:
        raise typer.Exit(1)


@native_app.command("status")
def native_status() -> None:
    """Show which native libraries are installed, where, and at which release."""
    from dictum.native_installer import install_status, native_root

    console.print(f"[dim]native root:[/dim] {native_root()}")
    status = install_status()
    for name, info in status.items():
        if info.get("error"):
            console.print(f"[red]✗[/red] {name.value}: {info['error']}")
            continue
        if info["installed"]:
            console.print(
                f"[green]✓[/green] {name.value} {info['release']} "
                f"({info['variant']}) → {info['path']}"
            )
        else:
            console.print(
                f"[yellow]·[/yellow] {name.value} {info['release']} "
                f"({info['variant']}) — not installed"
            )


# ──────────────────────────────────────────────
# service (systemd user unit install/management)
# ──────────────────────────────────────────────

service_app = typer.Typer(
    no_args_is_help=True,
    help="Install and manage the dictum systemd user service.",
)
app.add_typer(service_app, name="service")


def _user_unit_dir() -> Path:
    """Return the systemd user unit directory, creating it if needed."""
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        unit_dir = Path(base) / "systemd" / "user"
    else:
        unit_dir = Path.home() / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    return unit_dir


def _bundled_unit_path() -> Path:
    """Return the path to the unit file bundled inside the dictum package."""
    from importlib.resources import files

    p = Path(str(files("dictum").joinpath("data", "dictum.service")))
    if not p.exists():
        _die(
            "Bundled unit file dictum/data/dictum.service not found. "
            "Your dictum install may be incomplete."
        )
    return p


def _run_systemctl(*args: str) -> tuple[int, str, str]:
    """Run `systemctl --user <args>`; return (rc, stdout, stderr)."""
    cmd = ["systemctl", "--user", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stdout, proc.stderr


@service_app.command("install")
def service_install(
    now: Annotated[
        bool,
        typer.Option("--now", help="Also start the service immediately after enabling."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing unit file."),
    ] = False,
) -> None:
    """Install the dictum systemd user service (enabled, not started).

    Copies the bundled unit file into ~/.config/systemd/user/, runs
    `systemctl --user daemon-reload`, and enables the unit. By default the
    service is enabled (starts at login) but not started; pass --now to
    start it immediately.

    User units cannot start at boot — they start at graphical/login session.
    """
    import shutil

    # Sanity-check systemctl is available before touching files.
    if shutil.which("systemctl") is None:
        _die("systemctl not found on PATH. systemd user services are unavailable.")

    unit_dir = _user_unit_dir()
    dest = unit_dir / "dictum.service"

    if dest.exists() and not force:
        console.print(
            f"[yellow]·[/yellow] Unit file already exists at {dest} (use --force to overwrite)"
        )
    else:
        src = _bundled_unit_path()
        shutil.copyfile(src, dest)
        console.print(f"[green]✓[/green] Installed unit → {dest}")

    # daemon-reload
    rc, _out, err = _run_systemctl("daemon-reload")
    if rc != 0:
        _die(f"systemctl --user daemon-reload failed: {err.strip()}")
    console.print("[dim]daemon-reload done[/dim]")

    # enable (do not start unless --now)
    enable_args = ["enable"]
    if now:
        enable_args.append("--now")
    rc, out, err = _run_systemctl(*enable_args, "dictum.service")
    if rc != 0:
        _die(f"systemctl --user enable failed: {err.strip() or out.strip()}")
    if now:
        console.print("[green]✓[/green] dictum.service enabled and started")
    else:
        console.print("[green]✓[/green] dictum.service enabled (will start at next login)")
        console.print("[dim]  Start now with: systemctl --user start dictum[/dim]")


@service_app.command("uninstall")
def service_uninstall() -> None:
    """Disable and remove the dictum systemd user service."""
    import shutil

    if shutil.which("systemctl") is None:
        _die("systemctl not found on PATH.")

    # disable (no-op if not enabled)
    rc, _out, _err = _run_systemctl("disable", "dictum.service")
    if rc != 0:
        # Not enabled is fine; surface other errors.
        if "not loaded" not in _err and "No such file" not in _err:
            _die(f"systemctl --user disable failed: {_err.strip()}")
        console.print("[dim]·[/dim] dictum.service was not enabled")
    else:
        console.print("[green]✓[/green] dictum.service disabled")

    unit_path = _user_unit_dir() / "dictum.service"
    if unit_path.exists():
        unit_path.unlink()
        console.print(f"[green]✓[/green] Removed {unit_path}")
    else:
        console.print("[dim]·[/dim] No unit file to remove")

    rc, _out, err = _run_systemctl("daemon-reload")
    if rc != 0:
        _die(f"systemctl --user daemon-reload failed: {err.strip()}")
    console.print("[dim]daemon-reload done[/dim]")


@service_app.command("status")
def service_status() -> None:
    """Show whether the dictum systemd user service is installed and running."""
    import shutil

    if shutil.which("systemctl") is None:
        _die("systemctl not found on PATH.")

    unit_path = _user_unit_dir() / "dictum.service"
    installed = unit_path.exists()
    console.print(
        f"{'[green]✓[/green]' if installed else '[yellow]·[/yellow]'} "
        f"unit file: {unit_path} "
        f"{'(installed)' if installed else '(not installed)'}"
    )

    rc, out, err = _run_systemctl("is-enabled", "dictum.service")
    enabled_state = out.strip() or err.strip() or "unknown"
    if rc != 0 and enabled_state in ("disabled", "enabled", "static"):
        # is-enabled returns rc=1 for "disabled"; treat as informational
        pass
    console.print(f"  enabled: {enabled_state}")

    rc, out, _err = _run_systemctl("is-active", "dictum.service")
    active_state = out.strip() or "unknown"
    console.print(f"  active:  {active_state}")


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
