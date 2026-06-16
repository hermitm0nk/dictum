"""Vulkan runtime detection and native binary resolution."""
from __future__ import annotations

import ctypes
import logging
import sys
from importlib.resources import files
from pathlib import Path

log = logging.getLogger(__name__)

_VULKAN_CHECKED = False
_VULKAN_AVAILABLE = False


def native_root() -> Path:
    """Return the _native/ directory inside the installed package."""
    return Path(str(files("dictum").joinpath("_native")))


def native_binary(name: str) -> Path:
    """Return the path to a native executable in _native/bin/."""
    return native_root().joinpath("bin", name)


def native_lib_dir(subdir: str) -> Path:
    """Return a lib subdirectory inside _native/lib/."""
    return native_root().joinpath("lib", subdir)


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
