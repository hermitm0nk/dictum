"""Tests for the `dictum service` CLI subcommand.

All systemctl invocations are mocked via a fake `_run_systemctl` patched
onto `dictum.cli`. No real systemd session is required.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dictum import cli

runner = CliRunner()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _xdg_config_env(monkeypatch, tmp_path: Path) -> Path:
    """Point XDG_CONFIG_HOME at tmp_path; returns the expected unit dir."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path / "systemd" / "user"


def _fake_systemctl_success(*args: str) -> tuple[int, str, str]:
    """A stand-in for cli._run_systemctl that always succeeds."""
    if "is-enabled" in args:
        return 0, "enabled\n", ""
    if "is-active" in args:
        return 0, "active\n", ""
    return 0, "", ""


def _patch_systemctl(monkeypatch, handler=_fake_systemctl_success) -> list[list[str]]:
    """Replace cli._run_systemctl with `handler`. Returns a calls log."""
    calls: list[list[str]] = []

    def fake(*args: str) -> tuple[int, str, str]:
        calls.append(list(args))
        return handler(*args)

    monkeypatch.setattr(cli, "_run_systemctl", fake)
    return calls


def _patch_which(monkeypatch, exists: bool = True) -> None:
    """Make shutil.which find (or not find) systemctl."""
    monkeypatch.setattr(
        cli.shutil,
        "which",
        lambda name: "/usr/bin/systemctl" if exists and name == "systemctl" else None,
    )


# Need to patch shutil in the cli module's view, since cli.py imports shutil
# lazily inside each command. We patch the `shutil` module directly.
@pytest.fixture(autouse=True)
def _patch_shutil_which(monkeypatch):
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: "/usr/bin/systemctl" if name == "systemctl" else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# install
# ─────────────────────────────────────────────────────────────────────────────


def test_service_install_copies_unit_and_enables(monkeypatch, tmp_path: Path) -> None:
    unit_dir = _xdg_config_env(monkeypatch, tmp_path)
    calls = _patch_systemctl(monkeypatch)

    result = runner.invoke(cli.app, ["service", "install"])

    assert result.exit_code == 0, result.stderr
    assert (unit_dir / "dictum.service").exists()
    assert (unit_dir / "dictum.service").read_text().startswith("[Unit]")

    # daemon-reload then enable (no --now → no --now in enable args)
    systemctl_cmds = [c for c in calls if c and c[0] != "__"]
    assert ["daemon-reload"] in systemctl_cmds
    assert ["enable", "dictum.service"] in systemctl_cmds
    assert ["enable", "--now", "dictum.service"] not in systemctl_cmds


def test_service_install_now_starts(monkeypatch, tmp_path: Path) -> None:
    _xdg_config_env(monkeypatch, tmp_path)
    calls = _patch_systemctl(monkeypatch)

    result = runner.invoke(cli.app, ["service", "install", "--now"])

    assert result.exit_code == 0, result.stderr
    assert ["enable", "--now", "dictum.service"] in calls


def test_service_install_force_overwrites_existing(monkeypatch, tmp_path: Path) -> None:
    unit_dir = _xdg_config_env(monkeypatch, tmp_path)
    unit_dir.mkdir(parents=True, exist_ok=True)
    (unit_dir / "dictum.service").write_text("# stale\n")
    _patch_systemctl(monkeypatch)

    result = runner.invoke(cli.app, ["service", "install", "--force"])

    assert result.exit_code == 0, result.stderr
    assert (unit_dir / "dictum.service").read_text().startswith("[Unit]")


def test_service_install_without_force_keeps_existing(monkeypatch, tmp_path: Path) -> None:
    unit_dir = _xdg_config_env(monkeypatch, tmp_path)
    unit_dir.mkdir(parents=True, exist_ok=True)
    (unit_dir / "dictum.service").write_text("# my custom unit\n")
    _patch_systemctl(monkeypatch)

    result = runner.invoke(cli.app, ["service", "install"])

    assert result.exit_code == 0, result.stderr
    # Existing file is preserved; not overwritten without --force.
    assert (unit_dir / "dictum.service").read_text() == "# my custom unit\n"


def test_service_install_daemon_reload_failure_exits(monkeypatch, tmp_path: Path) -> None:
    _xdg_config_env(monkeypatch, tmp_path)

    def failing(*args: str) -> tuple[int, str, str]:
        if "daemon-reload" in args:
            return 1, "", "Failed to connect to bus\n"
        return 0, "", ""

    _patch_systemctl(monkeypatch, failing)

    result = runner.invoke(cli.app, ["service", "install"])

    assert result.exit_code != 0
    assert "daemon-reload failed" in result.stderr


def test_service_install_no_systemctl_exits(monkeypatch, tmp_path: Path) -> None:
    _xdg_config_env(monkeypatch, tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    _patch_systemctl(monkeypatch)

    result = runner.invoke(cli.app, ["service", "install"])

    assert result.exit_code != 0
    assert "systemctl not found" in result.stderr


def test_service_install_uses_xdg_config_home(monkeypatch, tmp_path: Path) -> None:
    unit_dir = _xdg_config_env(monkeypatch, tmp_path)
    _patch_systemctl(monkeypatch)

    runner.invoke(cli.app, ["service", "install"])

    assert (unit_dir / "dictum.service").exists()


def test_service_install_falls_back_to_home_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(cli.Path, "home", lambda: tmp_path)
    _patch_systemctl(monkeypatch)

    runner.invoke(cli.app, ["service", "install"])

    assert (tmp_path / ".config" / "systemd" / "user" / "dictum.service").exists()


# ─────────────────────────────────────────────────────────────────────────────
# uninstall
# ─────────────────────────────────────────────────────────────────────────────


def test_service_uninstall_removes_unit_and_disables(monkeypatch, tmp_path: Path) -> None:
    unit_dir = _xdg_config_env(monkeypatch, tmp_path)
    unit_dir.mkdir(parents=True, exist_ok=True)
    (unit_dir / "dictum.service").write_text("[Unit]\n")
    calls = _patch_systemctl(monkeypatch)

    result = runner.invoke(cli.app, ["service", "uninstall"])

    assert result.exit_code == 0, result.stderr
    assert not (unit_dir / "dictum.service").exists()
    assert ["disable", "dictum.service"] in calls
    assert ["daemon-reload"] in calls


def test_service_uninstall_without_unit_is_idempotent(monkeypatch, tmp_path: Path) -> None:
    _xdg_config_env(monkeypatch, tmp_path)
    _patch_systemctl(monkeypatch)

    result = runner.invoke(cli.app, ["service", "uninstall"])

    assert result.exit_code == 0, result.stderr


def test_service_uninstall_disable_not_loaded_is_tolerated(monkeypatch, tmp_path: Path) -> None:
    _xdg_config_env(monkeypatch, tmp_path)

    def failing_disable(*args: str) -> tuple[int, str, str]:
        if "disable" in args:
            return 1, "", "Failed to disable unit: Unit dictum.service not loaded.\n"
        return 0, "", ""

    _patch_systemctl(monkeypatch, failing_disable)

    result = runner.invoke(cli.app, ["service", "uninstall"])

    assert result.exit_code == 0, result.stderr


# ─────────────────────────────────────────────────────────────────────────────
# status
# ─────────────────────────────────────────────────────────────────────────────


def test_service_status_reports_installed_and_enabled(monkeypatch, tmp_path: Path) -> None:
    unit_dir = _xdg_config_env(monkeypatch, tmp_path)
    unit_dir.mkdir(parents=True, exist_ok=True)
    (unit_dir / "dictum.service").write_text("[Unit]\n")
    _patch_systemctl(monkeypatch)

    result = runner.invoke(cli.app, ["service", "status"])

    assert result.exit_code == 0, result.stderr
    # console = Console(stderr=True), so all dictum output is on stderr.
    assert "unit file" in result.stderr
    assert "installed" in result.stderr
    assert "enabled" in result.stderr
    assert "active" in result.stderr


def test_service_status_reports_not_installed(monkeypatch, tmp_path: Path) -> None:
    _xdg_config_env(monkeypatch, tmp_path)
    _patch_systemctl(monkeypatch)

    result = runner.invoke(cli.app, ["service", "status"])

    assert result.exit_code == 0, result.stderr
    assert "not installed" in result.stderr


def test_service_status_handles_disabled_is_enabled_rc(monkeypatch, tmp_path: Path) -> None:
    unit_dir = _xdg_config_env(monkeypatch, tmp_path)
    unit_dir.mkdir(parents=True, exist_ok=True)
    (unit_dir / "dictum.service").write_text("[Unit]\n")

    def is_enabled_disabled(*args: str) -> tuple[int, str, str]:
        if "is-enabled" in args:
            return 1, "disabled\n", ""
        if "is-active" in args:
            return 3, "inactive\n", ""
        return 0, "", ""

    _patch_systemctl(monkeypatch, is_enabled_disabled)

    result = runner.invoke(cli.app, ["service", "status"])

    assert result.exit_code == 0, result.stderr
    assert "disabled" in result.stderr
    assert "inactive" in result.stderr


# ─────────────────────────────────────────────────────────────────────────────
# bundled unit file is present in the package
# ─────────────────────────────────────────────────────────────────────────────


def test_bundled_unit_file_exists_in_package() -> None:
    from importlib.resources import files

    p = Path(str(files("dictum").joinpath("data", "dictum.service")))
    assert p.exists(), f"Bundled unit file missing at {p}"
    text = p.read_text()
    assert "[Unit]" in text
    assert "Description=Dictum voice dictation daemon" in text
    assert "ExecStart=" in text
