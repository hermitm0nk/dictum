"""Native binary resolution, Vulkan probe, and subprocess env helpers.

The native binaries (llama-server, parakeet-main) are no longer bundled in
the wheel. They are downloaded on first run by `dictum.native_installer`
into $XDG_DATA_HOME/dictum/native/<lib>/<release>/.

This module keeps the legacy `native_binary()` / `native_lib_dir()` API
that `asr.py` and `llm_local.py` already use, but resolves those paths
against the installer's layout. The `DICTUM_PARAKEET_BIN` and
`DICTUM_LLM_BIN` env vars continue to override the default lookup.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from pathlib import Path

from dictum.native_installer import (
    NativeLib,
    NativeVariant,
    ensure_installed,
    installed_binary,
    lib_install_dir,
    resolve_asset,
)

log = logging.getLogger(__name__)

_VULKAN_CHECKED = False
_VULKAN_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Layout resolution
# ─────────────────────────────────────────────────────────────────────────────


def native_root() -> Path:
    """Root directory for installed native libraries.

    Same as `dictum.native_installer.native_root()`: honors $DICTUM_NATIVE_DIR,
    otherwise $XDG_DATA_HOME/dictum/native, otherwise
    ~/.local/share/dictum/native.
    """
    from dictum.native_installer import native_root as _nr

    return _nr()


def native_binary(name: str) -> Path:
    """Return the path to a native executable managed by the installer.

    `name` is the dictum-internal binary name:
      - "llama-server"   → llama.cpp's llama-server
      - "parakeet-main"  → CrispASR's crispasr (renamed at install time)

    The path is returned even if the binary is not yet installed; callers
    that need it to exist should use `ensure_native_binary()` instead, or
    check `.exists()`.
    """
    if name == "llama-server":
        spec = resolve_asset(NativeLib.LLAMA, NativeVariant.VULKAN)
        return lib_install_dir(NativeLib.LLAMA, spec.release) / "llama-server"
    if name == "parakeet-main":
        spec = resolve_asset(NativeLib.CRISPASR, NativeVariant.VULKAN)
        return lib_install_dir(NativeLib.CRISPASR, spec.release) / "parakeet-main"
    # Unknown name: return a sensible default path so callers can produce a
    # clear "not found" error rather than a KeyError.
    return native_root() / "bin" / name


def native_lib_dir(subdir: str) -> Path:
    """Return a lib subdirectory inside the native root.

    For backwards compatibility with the old bundled-wheel layout, callers
    may pass "llama" or "crispasr". We map those to the installer's
    per-release directories.
    """
    if subdir in ("llama", "crispasr"):
        lib = NativeLib(subdir)
        spec = resolve_asset(lib, NativeVariant.VULKAN)
        return lib_install_dir(lib, spec.release)
    return native_root() / "lib" / subdir


def ensure_native_binary(name: str) -> Path:
    """Ensure the named native binary is installed; download if missing.

    Returns the absolute path to the binary. Raises if the platform is
    unsupported or the download fails.

    `name` follows the same convention as `native_binary()`.
    """
    if name == "llama-server":
        return ensure_installed(NativeLib.LLAMA, NativeVariant.VULKAN)
    if name == "parakeet-main":
        return ensure_installed(NativeLib.CRISPASR, NativeVariant.VULKAN)
    raise ValueError(f"Unknown native binary: {name!r}")


def native_env_for(name: str) -> dict[str, str]:
    """Return an environment dict with LD_LIBRARY_PATH set for `name`'s libs.

    Pre-built binaries already find their sibling .so files via $ORIGIN
    RUNPATH (llama-server) or are statically linked (parakeet-main), so
    this is a belt-and-suspenders safety wrapper. It does no harm when
    applied and future-proofs against upstream RPATH changes.
    """
    env = dict(os.environ)
    if name == "llama-server":
        lib_dir = native_lib_dir("llama")
    elif name == "parakeet-main":
        lib_dir = native_lib_dir("crispasr")
    else:
        return env

    if not lib_dir.exists():
        return env

    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else str(lib_dir)
    return env


# ─────────────────────────────────────────────────────────────────────────────
# Vulkan probe (unchanged behavior)
# ─────────────────────────────────────────────────────────────────────────────


def vulkan_available() -> bool:
    """Check if Vulkan loader is present and at least one ICD is available.

    Uses ctypes dlopen to probe the Vulkan loader without importing
    any Vulkan Python bindings.  Caches the result for the process lifetime.
    """
    global _VULKAN_CHECKED, _VULKAN_AVAILABLE
    if _VULKAN_CHECKED:
        return _VULKAN_AVAILABLE
    _VULKAN_CHECKED = True

    if sys.platform != "linux":
        log.info("Vulkan probe skipped (platform=%s)", sys.platform)
        return False

    for soname in ("libvulkan.so.1", "libvulkan.so"):
        try:
            lib = ctypes.CDLL(soname)
            # vkCreateInstance is always exported; use it as a presence test.
            if hasattr(lib, "vkCreateInstance"):
                _VULKAN_AVAILABLE = True
                log.info("Vulkan loader found (%s)", soname)
                return True
        except OSError:
            continue

    log.warning(
        "Vulkan not available — GPU acceleration disabled. "
        "Install the Vulkan loader (libvulkan1/vulkan-loader) and a "
        "compatible GPU driver for LLM acceleration."
    )
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: pre-install check used by daemon/CLI to short-circuit downloads
# ─────────────────────────────────────────────────────────────────────────────


def is_native_installed(name: str) -> bool:
    """True if the named native binary is already on disk (no download)."""
    if name == "llama-server":
        return installed_binary(NativeLib.LLAMA, NativeVariant.VULKAN) is not None
    if name == "parakeet-main":
        return installed_binary(NativeLib.CRISPASR, NativeVariant.VULKAN) is not None
    return native_binary(name).exists()
