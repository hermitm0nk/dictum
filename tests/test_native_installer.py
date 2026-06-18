"""Tests for dictum.native_installer and dictum.accel.

These tests never touch the network: download is monkeypatched, and the
SHA-256 digests used by the install flow are fixtures, not the real
upstream pins.
"""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
from pathlib import Path

import pytest

from dictum import accel
from dictum import native_installer as ni
from dictum.native_installer import (
    AssetNotAvailableError,
    AssetSpec,
    ExtractError,
    HashMismatchError,
    NativeLib,
    NativeVariant,
    PlatformNotSupportedError,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers: build in-memory tarballs and sync the platform key.
# ─────────────────────────────────────────────────────────────────────────────


def _make_tarball(top_dir: str, entries: dict[str, bytes | str]) -> bytes:
    """Build an in-memory .tar.gz with a single top dir and given entries.

    Entries are path -> bytes (file content) or path -> str (symlink target,
    prefixed with 'symlink:').
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        # Top dir entry.
        ti = tarfile.TarInfo(name=top_dir)
        ti.type = tarfile.DIRTYPE
        ti.mode = 0o755
        tf.addfile(ti)
        for rel, value in entries.items():
            name = f"{top_dir}/{rel}"
            if isinstance(value, str) and value.startswith("symlink:"):
                target = value[len("symlink:") :]
                ti = tarfile.TarInfo(name=name)
                ti.type = tarfile.SYMTYPE
                ti.linkname = target
                ti.mode = 0o777
                tf.addfile(ti)
            else:
                assert isinstance(value, (bytes, bytearray))
                data = bytes(value)
                ti = tarfile.TarInfo(name=name)
                ti.size = len(data)
                ti.mode = 0o755
                tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def _force_platform(monkeypatch, system: str, machine: str) -> None:
    """Make native_installer see the requested platform."""
    monkeypatch.setattr(ni.platform, "system", lambda: system)
    monkeypatch.setattr(ni.platform, "machine", lambda: machine)


def _xdg_env(monkeypatch, tmp_path: Path) -> Path:
    """Point XDG vars and DICTUM_NATIVE_DIR at tmp_path; returns native_root."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    data_home = tmp_path / "data"
    data_home.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.delenv("DICTUM_NATIVE_DIR", raising=False)
    return data_home / "dictum" / "native"


# ─────────────────────────────────────────────────────────────────────────────
# Platform key + asset resolution
# ─────────────────────────────────────────────────────────────────────────────


def test_platform_key_normalizes_x86_64_aliases(monkeypatch) -> None:
    _force_platform(monkeypatch, "Linux", "x86_64")
    assert ni._platform_key() == ("linux", "x86_64")

    _force_platform(monkeypatch, "linux", "amd64")
    assert ni._platform_key() == ("linux", "x86_64")


def test_platform_key_normalizes_arm64_aliases(monkeypatch) -> None:
    _force_platform(monkeypatch, "Linux", "aarch64")
    assert ni._platform_key() == ("linux", "arm64")

    _force_platform(monkeypatch, "linux", "arm64")
    assert ni._platform_key() == ("linux", "arm64")


def test_platform_key_rejects_non_linux(monkeypatch) -> None:
    _force_platform(monkeypatch, "Darwin", "x86_64")
    with pytest.raises(PlatformNotSupportedError):
        ni._platform_key()


def test_platform_key_rejects_unknown_arch(monkeypatch) -> None:
    _force_platform(monkeypatch, "linux", "riscv64")
    with pytest.raises(PlatformNotSupportedError):
        ni._platform_key()


def test_resolve_asset_llama_vulkan_x86_64(monkeypatch) -> None:
    _force_platform(monkeypatch, "linux", "x86_64")
    spec = ni.resolve_asset(NativeLib.LLAMA, NativeVariant.VULKAN)
    assert spec.release == ni.LLAMA_CPP_RELEASE
    assert spec.filename == "llama-b9699-bin-ubuntu-vulkan-x64.tar.gz"
    assert spec.binary_name == "llama-server"
    assert spec.sha256 == ("4e4ce9582ab43706eff9528896e841e3f239f513f8fe9421298f1b1156ad852a")


def test_resolve_asset_crispasr_vulkan_x86_64(monkeypatch) -> None:
    _force_platform(monkeypatch, "linux", "x86_64")
    spec = ni.resolve_asset(NativeLib.CRISPASR, NativeVariant.VULKAN)
    assert spec.release == ni.CRISPASR_RELEASE
    assert spec.filename == "crispasr-linux-x86_64-vulkan.tar.gz"
    assert spec.binary_name == "crispasr"
    assert spec.binary_rename == "parakeet-main"


def test_resolve_asset_crispasr_arm64_vulkan_falls_back_to_cpu(monkeypatch, caplog) -> None:
    _force_platform(monkeypatch, "linux", "arm64")
    spec = ni.resolve_asset(NativeLib.CRISPASR, NativeVariant.VULKAN)
    # No arm64 Vulkan asset; falls back to the CPU variant.
    assert spec.filename == "crispasr-linux-arm64.tar.gz"
    assert "falling back to cpu" in caplog.text.lower()


def test_resolve_asset_llama_cpu_x86_64_available(monkeypatch) -> None:
    _force_platform(monkeypatch, "linux", "x86_64")
    spec = ni.resolve_asset(NativeLib.LLAMA, NativeVariant.CPU)
    assert spec.filename == "llama-b9699-bin-ubuntu-x64.tar.gz"


def test_resolve_asset_unavailable_raises(monkeypatch) -> None:
    _force_platform(monkeypatch, "linux", "arm64")
    # llama.cpp has an arm64 Vulkan asset, but no arm64 CPU asset is pinned.
    with pytest.raises(AssetNotAvailableError):
        ni.resolve_asset(NativeLib.LLAMA, NativeVariant.CPU)


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────


def test_native_root_honors_override(monkeypatch, tmp_path: Path) -> None:
    custom = tmp_path / "custom-native"
    monkeypatch.setenv("DICTUM_NATIVE_DIR", str(custom))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert ni.native_root() == custom


def test_native_root_uses_xdg_data_home(monkeypatch, tmp_path: Path) -> None:
    _xdg_env(monkeypatch, tmp_path)
    assert ni.native_root() == tmp_path / "data" / "dictum" / "native"


def test_native_root_falls_back_to_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DICTUM_NATIVE_DIR", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(ni.Path, "home", lambda: tmp_path)
    assert ni.native_root() == tmp_path / ".local" / "share" / "dictum" / "native"


def test_lib_install_dir_layout(monkeypatch, tmp_path: Path) -> None:
    _xdg_env(monkeypatch, tmp_path)
    p = ni.lib_install_dir(NativeLib.LLAMA, "b9699")
    assert p == tmp_path / "data" / "dictum" / "native" / "llama" / "b9699"


# ─────────────────────────────────────────────────────────────────────────────
# Marker file
# ─────────────────────────────────────────────────────────────────────────────


def test_read_marker_missing_returns_none(tmp_path: Path) -> None:
    assert ni._read_marker(tmp_path / ".installed") is None


def test_read_marker_parses_valid(tmp_path: Path) -> None:
    p = tmp_path / ".installed"
    p.write_text("b9699\nabcd\nfile.tar.gz\n")
    assert ni._read_marker(p) == ("b9699", "abcd", "file.tar.gz")


def test_read_marker_rejects_malformed(tmp_path: Path) -> None:
    p = tmp_path / ".installed"
    p.write_text("only-one-line\n")
    assert ni._read_marker(p) is None


# ─────────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────────


def test_single_top_dir_detects_one(monkeypatch) -> None:
    blob = _make_tarball("pkg-1.0", {"bin/foo": b"hi", "lib/lib.so": b"x"})
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        assert ni._single_top_dir(tf) == "pkg-1.0"


def test_single_top_dir_none_for_multi(monkeypatch) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for top in ("a", "b"):
            ti = tarfile.TarInfo(name=top)
            ti.type = tarfile.DIRTYPE
            tf.addfile(ti)
    with tarfile.open(fileobj=io.BytesIO(buf.getvalue()), mode="r:gz") as tf:
        assert ni._single_top_dir(tf) is None


def test_extract_flat_strips_top_dir(tmp_path: Path) -> None:
    blob = _make_tarball(
        "pkg-1.0",
        {"bin/foo": b"#!/bin/sh\necho hi\n", "lib/lib.so": b"\x7fELF fake"},
    )
    dest = tmp_path / "out"
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        ni._extract_flat(tf, dest, expected_top="pkg-1.0")
    assert (dest / "bin" / "foo").read_bytes() == b"#!/bin/sh\necho hi\n"
    assert (dest / "lib" / "lib.so").read_bytes() == b"\x7fELF fake"
    assert not (dest / "pkg-1.0").exists()


def test_extract_flat_preserves_symlinks(tmp_path: Path) -> None:
    blob = _make_tarball(
        "pkg",
        {
            "lib/lib.so.0.1": b"real-content",
            "lib/lib.so.0": "symlink:lib.so.0.1",
            "lib/lib.so": "symlink:lib.so.0",
        },
    )
    dest = tmp_path / "out"
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        ni._extract_flat(tf, dest, expected_top="pkg")
    assert (dest / "lib" / "lib.so.0.1").read_bytes() == b"real-content"
    assert (dest / "lib" / "lib.so.0").is_symlink()
    assert os.readlink(dest / "lib" / "lib.so.0") == "lib.so.0.1"
    assert (dest / "lib" / "lib.so").is_symlink()
    assert os.readlink(dest / "lib" / "lib.so") == "lib.so.0"


def test_extract_flat_raises_on_multi_top_without_expected(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for top in ("a", "b"):
            ti = tarfile.TarInfo(name=top)
            ti.type = tarfile.DIRTYPE
            tf.addfile(ti)
    with tarfile.open(fileobj=io.BytesIO(buf.getvalue()), mode="r:gz") as tf:
        with pytest.raises(ExtractError):
            ni._extract_flat(tf, tmp_path / "out", expected_top=None)


# ─────────────────────────────────────────────────────────────────────────────
# SHA-256
# ─────────────────────────────────────────────────────────────────────────────


def test_sha256_file_matches_known(tmp_path: Path) -> None:
    data = b"dictum-test-sha\n"
    p = tmp_path / "f"
    p.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert ni._sha256_file(p) == expected


# ─────────────────────────────────────────────────────────────────────────────
# install_lib end-to-end (download monkeypatched)
# ─────────────────────────────────────────────────────────────────────────────


def _fake_asset(tmp_path: Path, entries: dict[str, bytes | str]) -> tuple[AssetSpec, bytes, str]:
    """Build a fake tarball + matching AssetSpec for testing install_lib."""
    blob = _make_tarball("fakepkg", entries)
    sha = hashlib.sha256(blob).hexdigest()
    spec = AssetSpec(
        filename="fake.tar.gz",
        sha256=sha,
        binary_name="crispasr",
        binary_rename="parakeet-main",
        top_dir="fakepkg",
        release="vFAKE",
    )
    return spec, blob, sha


def _patch_install(monkeypatch, lib: NativeLib, spec: AssetSpec, blob: bytes) -> None:
    """Inject a fake asset into the asset table and patch _download + URL."""
    # Pin the fake asset for linux/x86_64.
    monkeypatch.setattr(ni.platform, "system", lambda: "linux")
    monkeypatch.setattr(ni.platform, "machine", lambda: "x86_64")
    fake_key = (lib, "linux", "x86_64", NativeVariant.VULKAN)
    monkeypatch.setitem(ni._ASSETS, fake_key, spec)
    release_attr = "LLAMA_CPP_RELEASE" if lib is NativeLib.LLAMA else "CRISPASR_RELEASE"
    monkeypatch.setattr(ni, release_attr, spec.release)

    # Patch _download to write our blob instead of hitting the network.
    def fake_download(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)

    monkeypatch.setattr(ni, "_download", fake_download)


def test_install_lib_downloads_extracts_and_renames(monkeypatch, tmp_path: Path) -> None:
    _xdg_env(monkeypatch, tmp_path)
    spec, blob, _sha = _fake_asset(
        tmp_path,
        {"crispasr": b"\x7fELF fake binary", "README": b"docs"},
    )
    _patch_install(monkeypatch, NativeLib.CRISPASR, spec, blob)

    binary = ni.install_lib(NativeLib.CRISPASR, NativeVariant.VULKAN)

    install_dir = ni.lib_install_dir(NativeLib.CRISPASR, spec.release)
    assert binary == install_dir / "parakeet-main"
    assert binary.exists()
    assert binary.read_bytes() == b"\x7fELF fake binary"
    # Original crispasr name is gone; rename happened.
    assert not (install_dir / "crispasr").exists()
    # Marker file recorded release + sha + filename.
    marker = (install_dir / ni.INSTALLED_MARKER).read_text().strip().split("\n")
    assert marker[0] == "vFAKE"
    assert marker[1] == spec.sha256
    assert marker[2] == "fake.tar.gz"
    # Binary is executable.
    assert os.access(binary, os.X_OK)


def test_install_lib_idempotent_second_call_noop(monkeypatch, tmp_path: Path) -> None:
    _xdg_env(monkeypatch, tmp_path)
    spec, blob, _sha = _fake_asset(tmp_path, {"crispasr": b"bin"})
    _patch_install(monkeypatch, NativeLib.CRISPASR, spec, blob)

    ni.install_lib(NativeLib.CRISPASR, NativeVariant.VULKAN)

    # Patch _download to blow up; if it's called, the second install didn't
    # short-circuit on the marker.
    def boom(url: str, dest: Path) -> None:
        raise AssertionError("download should not run on second install")

    monkeypatch.setattr(ni, "_download", boom)

    binary = ni.install_lib(NativeLib.CRISPASR, NativeVariant.VULKAN)
    assert binary.exists()


def test_install_lib_force_reinstalls(monkeypatch, tmp_path: Path) -> None:
    _xdg_env(monkeypatch, tmp_path)
    spec, blob, _sha = _fake_asset(tmp_path, {"crispasr": b"new-bin"})
    _patch_install(monkeypatch, NativeLib.CRISPASR, spec, blob)

    # First install.
    ni.install_lib(NativeLib.CRISPASR, NativeVariant.VULKAN)
    # Corrupt the binary to prove force re-installs.
    binary = ni.lib_install_dir(NativeLib.CRISPASR, spec.release) / "parakeet-main"
    binary.write_bytes(b"corrupted")

    ni.install_lib(NativeLib.CRISPASR, NativeVariant.VULKAN, force=True)

    assert binary.read_bytes() == b"new-bin"


def test_install_lib_hash_mismatch_raises(monkeypatch, tmp_path: Path) -> None:
    _xdg_env(monkeypatch, tmp_path)
    # Asset spec claims a sha that won't match the blob we deliver.
    spec = AssetSpec(
        filename="fake.tar.gz",
        sha256="0" * 64,  # wrong
        binary_name="crispasr",
        binary_rename="parakeet-main",
        top_dir="fakepkg",
        release="vFAKE",
    )
    blob = _make_tarball("fakepkg", {"crispasr": b"bin"})
    _patch_install(monkeypatch, NativeLib.CRISPASR, spec, blob)

    with pytest.raises(HashMismatchError):
        ni.install_lib(NativeLib.CRISPASR, NativeVariant.VULKAN)


def test_install_lib_missing_binary_in_tarball_raises(monkeypatch, tmp_path: Path) -> None:
    _xdg_env(monkeypatch, tmp_path)
    spec = AssetSpec(
        filename="fake.tar.gz",
        sha256="x" * 64,  # placeholder; we'll compute below
        binary_name="crispasr",
        binary_rename="parakeet-main",
        top_dir="fakepkg",
        release="vFAKE",
    )
    # Tarball contains no "crispasr" binary.
    blob = _make_tarball("fakepkg", {"README": b"no binary here"})
    spec = AssetSpec(
        filename=spec.filename,
        sha256=hashlib.sha256(blob).hexdigest(),
        binary_name=spec.binary_name,
        binary_rename=spec.binary_rename,
        top_dir=spec.top_dir,
        release=spec.release,
    )
    _patch_install(monkeypatch, NativeLib.CRISPASR, spec, blob)

    with pytest.raises(ExtractError):
        ni.install_lib(NativeLib.CRISPASR, NativeVariant.VULKAN)


def test_install_lib_concurrent_second_call_sees_first(monkeypatch, tmp_path: Path) -> None:
    """Acquiring the flock after another process installed should no-op."""
    _xdg_env(monkeypatch, tmp_path)
    spec, blob, _sha = _fake_asset(tmp_path, {"crispasr": b"bin"})
    _patch_install(monkeypatch, NativeLib.CRISPASR, spec, blob)

    # First install creates marker + binary.
    ni.install_lib(NativeLib.CRISPASR, NativeVariant.VULKAN)

    # Patch is_installed to return False initially (simulating the pre-lock
    # check racing), then True after the lock is acquired. This verifies the
    # post-lock re-check.
    calls = {"n": 0}
    real_is_installed = ni.is_installed

    def flaky(lib: NativeLib, variant: NativeVariant = NativeVariant.VULKAN) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            return False  # pre-lock check
        return real_is_installed(lib, variant)  # post-lock check

    monkeypatch.setattr(ni, "is_installed", flaky)
    binary = ni.install_lib(NativeLib.CRISPASR, NativeVariant.VULKAN)
    assert binary.exists()


# ─────────────────────────────────────────────────────────────────────────────
# install_status / is_installed
# ─────────────────────────────────────────────────────────────────────────────


def test_is_installed_false_when_marker_missing(monkeypatch, tmp_path: Path) -> None:
    _xdg_env(monkeypatch, tmp_path)
    _force_platform(monkeypatch, "linux", "x86_64")
    assert ni.is_installed(NativeLib.LLAMA, NativeVariant.VULKAN) is False


def test_is_installed_true_after_install(monkeypatch, tmp_path: Path) -> None:
    _xdg_env(monkeypatch, tmp_path)
    spec, blob, _sha = _fake_asset(tmp_path, {"crispasr": b"bin"})
    _patch_install(monkeypatch, NativeLib.CRISPASR, spec, blob)
    ni.install_lib(NativeLib.CRISPASR, NativeVariant.VULKAN)
    assert ni.is_installed(NativeLib.CRISPASR, NativeVariant.VULKAN) is True


def test_is_installed_false_when_marker_sha_mismatches(monkeypatch, tmp_path: Path) -> None:
    _xdg_env(monkeypatch, tmp_path)
    spec, blob, _sha = _fake_asset(tmp_path, {"crispasr": b"bin"})
    _patch_install(monkeypatch, NativeLib.CRISPASR, spec, blob)
    ni.install_lib(NativeLib.CRISPASR, NativeVariant.VULKAN)

    # Tamper with the marker file.
    install_dir = ni.lib_install_dir(NativeLib.CRISPASR, spec.release)
    (install_dir / ni.INSTALLED_MARKER).write_text(f"{spec.release}\n{'0' * 64}\n{spec.filename}\n")
    assert ni.is_installed(NativeLib.CRISPASR, NativeVariant.VULKAN) is False


def test_install_status_reports_all_libs(monkeypatch, tmp_path: Path) -> None:
    _xdg_env(monkeypatch, tmp_path)
    _force_platform(monkeypatch, "linux", "x86_64")
    status = ni.install_status()
    assert NativeLib.LLAMA in status
    assert NativeLib.CRISPASR in status
    assert status[NativeLib.LLAMA]["installed"] is False
    assert status[NativeLib.CRISPASR]["installed"] is False


# ─────────────────────────────────────────────────────────────────────────────
# accel.native_env_for + native_binary
# ─────────────────────────────────────────────────────────────────────────────


def test_native_env_for_llama_sets_ld_library_path(monkeypatch, tmp_path: Path) -> None:
    _xdg_env(monkeypatch, tmp_path)
    _force_platform(monkeypatch, "linux", "x86_64")
    # Create the lib dir so the helper doesn't bail out.
    lib_dir = accel.native_lib_dir("llama")
    lib_dir.mkdir(parents=True)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    env = accel.native_env_for("llama-server")
    assert env["LD_LIBRARY_PATH"] == str(lib_dir)


def test_native_env_for_preserves_existing_ld_library_path(monkeypatch, tmp_path: Path) -> None:
    _xdg_env(monkeypatch, tmp_path)
    _force_platform(monkeypatch, "linux", "x86_64")
    lib_dir = accel.native_lib_dir("crispasr")
    lib_dir.mkdir(parents=True)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/other/libs")
    env = accel.native_env_for("parakeet-main")
    assert env["LD_LIBRARY_PATH"] == f"{lib_dir}:/opt/other/libs"


def test_native_env_for_unknown_name_returns_env_unchanged(monkeypatch, tmp_path: Path) -> None:
    _xdg_env(monkeypatch, tmp_path)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    env = accel.native_env_for("some-unknown-binary")
    assert "LD_LIBRARY_PATH" not in env


def test_native_env_for_missing_lib_dir_returns_env_unchanged(monkeypatch, tmp_path: Path) -> None:
    _xdg_env(monkeypatch, tmp_path)
    _force_platform(monkeypatch, "linux", "x86_64")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    # Lib dir does not exist (no install yet).
    env = accel.native_env_for("llama-server")
    assert "LD_LIBRARY_PATH" not in env


def test_native_binary_llama_server_path(monkeypatch, tmp_path: Path) -> None:
    _xdg_env(monkeypatch, tmp_path)
    _force_platform(monkeypatch, "linux", "x86_64")
    p = accel.native_binary("llama-server")
    assert p == tmp_path / "data" / "dictum" / "native" / "llama" / "b9699" / "llama-server"


def test_native_binary_parakeet_main_path(monkeypatch, tmp_path: Path) -> None:
    _xdg_env(monkeypatch, tmp_path)
    _force_platform(monkeypatch, "linux", "x86_64")
    p = accel.native_binary("parakeet-main")
    assert p == (tmp_path / "data" / "dictum" / "native" / "crispasr" / "v0.7.2" / "parakeet-main")


def test_native_lib_dir_llama_matches_installer_layout(monkeypatch, tmp_path: Path) -> None:
    _xdg_env(monkeypatch, tmp_path)
    _force_platform(monkeypatch, "linux", "x86_64")
    assert accel.native_lib_dir("llama") == (
        tmp_path / "data" / "dictum" / "native" / "llama" / "b9699"
    )


def test_is_native_installed_false_when_missing(monkeypatch, tmp_path: Path) -> None:
    _xdg_env(monkeypatch, tmp_path)
    _force_platform(monkeypatch, "linux", "x86_64")
    assert accel.is_native_installed("llama-server") is False
    assert accel.is_native_installed("parakeet-main") is False


def test_is_native_installed_true_after_install(monkeypatch, tmp_path: Path) -> None:
    _xdg_env(monkeypatch, tmp_path)
    spec, blob, _sha = _fake_asset(tmp_path, {"crispasr": b"bin"})
    _patch_install(monkeypatch, NativeLib.CRISPASR, spec, blob)
    ni.install_lib(NativeLib.CRISPASR, NativeVariant.VULKAN)
    assert accel.is_native_installed("parakeet-main") is True


def test_ensure_native_binary_unknown_name_raises(monkeypatch, tmp_path: Path) -> None:
    _xdg_env(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        accel.ensure_native_binary("not-a-real-binary")
