"""Download and cache pre-built native binaries (llama.cpp, CrispASR).

The wheel is pure-Python; on first run (or via `dictum native install`) we
fetch pinned upstream GitHub releases, verify their SHA-256, extract them
into $XDG_DATA_HOME/dictum/native/<lib>/<release>/, and write a `.installed`
marker so subsequent starts skip the download.

Layout (after install)::

    $XDG_DATA_HOME/dictum/native/
        llama/b9699/
            llama-server          # binary, has RUNPATH=$ORIGIN
            lib*.so               # sibling shared libraries
            .installed            # marker: "<release>\\n<sha256>\\n"
        crispasr/v0.7.2/
            parakeet-main         # binary, statically linked
            .installed

Concurrent installs are guarded by an flock on
$XDG_RUNTIME_DIR/dictum/native.lock so the daemon and a simultaneous
`dictum native install` cannot race.
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import platform
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import httpx
from tqdm import tqdm

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Pinned upstream releases
# ─────────────────────────────────────────────────────────────────────────────

LLAMA_CPP_RELEASE = "b9699"
CRISPASR_RELEASE = "v0.7.2"

LLAMA_CPP_REPO = "ggml-org/llama.cpp"
CRISPASR_REPO = "CrispStrobe/CrispASR"

GITHUB_RELEASE_BASE = "https://github.com"


class NativeLib(StrEnum):
    LLAMA = "llama"
    CRISPASR = "crispasr"


class NativeVariant(StrEnum):
    VULKAN = "vulkan"
    CPU = "cpu"


@dataclass(frozen=True)
class AssetSpec:
    """A pinned release asset: filename + expected SHA-256."""

    filename: str
    sha256: str
    binary_name: str
    """Name of the executable inside the tarball (after extraction)."""

    binary_rename: str | None = None
    """If set, the extracted binary is renamed to this."""

    top_dir: str | None = None
    """Expected top-level directory inside the tarball.

    If None, infer from the tarball itself (single top-level entry).
    """

    release: str = ""
    """Release tag this asset belongs to (filled in by resolver)."""


# Verified SHA-256 digests, sourced from the GitHub Releases API.
# These protect against partial downloads and tampering.
_ASSETS: dict[tuple[NativeLib, str, str, NativeVariant], AssetSpec] = {
    # llama.cpp b9699 — Linux Vulkan x86_64 (38 MB compressed, 110 MB extracted)
    (NativeLib.LLAMA, "linux", "x86_64", NativeVariant.VULKAN): AssetSpec(
        filename="llama-b9699-bin-ubuntu-vulkan-x64.tar.gz",
        sha256="4e4ce9582ab43706eff9528896e841e3f239f513f8fe9421298f1b1156ad852a",
        binary_name="llama-server",
        top_dir="llama-b9699",
        release=LLAMA_CPP_RELEASE,
    ),
    # llama.cpp b9699 — Linux Vulkan arm64 (32 MB)
    (NativeLib.LLAMA, "linux", "arm64", NativeVariant.VULKAN): AssetSpec(
        filename="llama-b9699-bin-ubuntu-vulkan-arm64.tar.gz",
        sha256="42b575e256e973da24901026e5c4123080aa6b18d3a81b43696ad4b4f79b0a36",
        binary_name="llama-server",
        top_dir="llama-b9699",
        release=LLAMA_CPP_RELEASE,
    ),
    # llama.cpp b9699 — Linux CPU x86_64 (16 MB) — fallback when no Vulkan ICD
    (NativeLib.LLAMA, "linux", "x86_64", NativeVariant.CPU): AssetSpec(
        filename="llama-b9699-bin-ubuntu-x64.tar.gz",
        sha256="22dbabb4c723d2fdc33a09b0c7b371f0dcd81d5e87083c2a2be96548db119011",
        binary_name="llama-server",
        top_dir="llama-b9699",
        release=LLAMA_CPP_RELEASE,
    ),
    # CrispASR v0.7.2 — Linux Vulkan x86_64 (46 MB, statically linked binary)
    (NativeLib.CRISPASR, "linux", "x86_64", NativeVariant.VULKAN): AssetSpec(
        filename="crispasr-linux-x86_64-vulkan.tar.gz",
        sha256="cfc7d259c89dcb28c633e3882fb314ed913d87d8a883d2c5cc2b287b3a46d95b",
        binary_name="crispasr",
        binary_rename="parakeet-main",
        top_dir="crispasr-linux-x86_64-vulkan",
        release=CRISPASR_RELEASE,
    ),
    # CrispASR v0.7.2 — Linux CPU x86_64 (10 MB) — fallback when no Vulkan ICD
    (NativeLib.CRISPASR, "linux", "x86_64", NativeVariant.CPU): AssetSpec(
        filename="crispasr-linux-x86_64.tar.gz",
        sha256="1a5bf9bab497e739e940f279c96c7110afd575b5c1234569be27a8c23d5267e2",
        binary_name="crispasr",
        binary_rename="parakeet-main",
        top_dir="crispasr-linux-x86_64",
        release=CRISPASR_RELEASE,
    ),
    # CrispASR v0.7.2 — Linux CPU arm64 (9 MB) — no arm64 Vulkan variant upstream
    (NativeLib.CRISPASR, "linux", "arm64", NativeVariant.CPU): AssetSpec(
        filename="crispasr-linux-arm64.tar.gz",
        sha256="4a9f729aab0b7dfabb56e3188cf5309884f0e6f0f0cbb43551f2f87faf4586a8",
        binary_name="crispasr",
        binary_rename="parakeet-main",
        top_dir="crispasr-linux-arm64",
        release=CRISPASR_RELEASE,
    ),
}

# CrispASR has no arm64 Vulkan build: silently fall back to CPU on aarch64.
_VARIANT_FALLBACK: dict[tuple[NativeLib, str, str, NativeVariant], NativeVariant] = {
    (NativeLib.CRISPASR, "linux", "arm64", NativeVariant.VULKAN): NativeVariant.CPU,
}


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────

INSTALLED_MARKER = ".installed"


def xdg_data_home() -> Path:
    """Return the base XDG data directory."""
    data_home = os.environ.get("XDG_DATA_HOME")
    if data_home:
        return Path(data_home).expanduser()
    return Path.home() / ".local" / "share"


def native_root() -> Path:
    """Root directory for all installed native libraries.

    Override with $DICTUM_NATIVE_DIR; defaults to $XDG_DATA_HOME/dictum/native.
    """
    override = os.environ.get("DICTUM_NATIVE_DIR")
    if override:
        return Path(override).expanduser()
    return xdg_data_home() / "dictum" / "native"


def lib_install_dir(lib: NativeLib, release: str) -> Path:
    """Directory where a given (lib, release) pair is extracted."""
    return native_root() / lib.value / release


def installed_binary_path(lib: NativeLib, release: str, spec: AssetSpec) -> Path:
    """Path to the binary after install (post-rename, if any)."""
    name = spec.binary_rename or spec.binary_name
    return lib_install_dir(lib, release) / name


def _lock_path() -> Path:
    """flock path under $XDG_RUNTIME_DIR, fallback to native_root()."""
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        d = Path(runtime) / "dictum"
    else:
        d = native_root().parent
    d.mkdir(parents=True, exist_ok=True)
    return d / "native.lock"


# ─────────────────────────────────────────────────────────────────────────────
# Asset resolution
# ─────────────────────────────────────────────────────────────────────────────


def _platform_key() -> tuple[str, str]:
    """Return (system, machine) normalized to our asset table keys.

    Only Linux is supported; raises on anything else.
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system != "linux":
        raise PlatformNotSupportedError(
            f"Pre-built native binaries are only available for Linux (got {system!r})."
        )

    # Normalize common machine aliases.
    if machine in ("x86_64", "amd64"):
        machine = "x86_64"
    elif machine in ("aarch64", "arm64"):
        machine = "arm64"
    else:
        raise PlatformNotSupportedError(f"No pre-built native binary for architecture {machine!r}.")

    return system, machine


def resolve_asset(
    lib: NativeLib,
    variant: NativeVariant = NativeVariant.VULKAN,
) -> AssetSpec:
    """Return the AssetSpec matching the current platform and chosen variant.

    Applies the documented fallbacks (e.g. CrispASR arm64 has no Vulkan build,
    so it falls back to CPU with a logged warning).
    """
    system, machine = _platform_key()
    key = (lib, system, machine, variant)

    if key in _ASSETS:
        return _ASSETS[key]

    fallback = _VARIANT_FALLBACK.get(key)
    if fallback is not None:
        log.warning(
            "No %s %s build for %s/%s; falling back to %s.",
            lib.value,
            variant.value,
            system,
            machine,
            fallback.value,
        )
        fb_key = (lib, system, machine, fallback)
        return _ASSETS[fb_key]

    raise AssetNotAvailableError(
        f"No pre-built {lib.value} {variant.value} asset for {system}/{machine}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────


class NativeInstallError(RuntimeError):
    """Base class for native-installer failures."""


class PlatformNotSupportedError(NativeInstallError):
    """The current platform has no pre-built asset."""


class AssetNotAvailableError(NativeInstallError):
    """The requested (lib, variant) has no asset for this platform."""


class DownloadError(NativeInstallError):
    """HTTP or network failure during download."""


class HashMismatchError(NativeInstallError):
    """Downloaded asset's SHA-256 did not match the pinned digest."""


class ExtractError(NativeInstallError):
    """Tarball extraction failed or layout was unexpected."""


# ─────────────────────────────────────────────────────────────────────────────
# Download + verify
# ─────────────────────────────────────────────────────────────────────────────


def _release_url(lib: NativeLib, release: str, filename: str) -> str:
    repo = LLAMA_CPP_REPO if lib is NativeLib.LLAMA else CRISPASR_REPO
    return f"{GITHUB_RELEASE_BASE}/{repo}/releases/download/{release}/{filename}"


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Stream `url` to `dest` with a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as resp:
            if resp.status_code != 200:
                raise DownloadError(f"HTTP {resp.status_code} for {url}")
            total = int(resp.headers.get("content-length", "0")) or None
            with (
                open(tmp, "wb") as f,
                tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                    leave=False,
                ) as bar,
            ):
                for chunk in resp.iter_bytes(chunk_size=1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
        os.replace(tmp, dest)
    except httpx.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"Network error downloading {url}: {exc}") from exc
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────────


def _single_top_dir(tf: tarfile.TarFile) -> str | None:
    """Return the single common top-level directory of a tarball, or None."""
    names = tf.getnames()
    if not names:
        return None
    tops = {n.split("/", 1)[0] for n in names if n}
    if len(tops) == 1:
        return next(iter(tops))
    return None


def _extract_flat(tf: tarfile.TarFile, dest: Path, expected_top: str | None) -> None:
    """Extract tarball, stripping the single top-level directory.

    All entries end up directly inside `dest`. Raises ExtractError if the
    layout is not a single top dir (when expected_top is None we infer it).
    """
    top = expected_top or _single_top_dir(tf)
    if top is None:
        raise ExtractError("Tarball has no single top-level directory; cannot strip safely.")

    dest.mkdir(parents=True, exist_ok=True)

    for member in tf.getmembers():
        name = member.name
        if name == top or not name.startswith(top + "/"):
            continue
        rel = name[len(top) + 1 :]
        if not rel:
            continue
        target = dest / rel

        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        elif member.issym() or member.islnk():
            # Preserve symlinks (e.g. libggml.so → libggml.so.0). Link target
            # is relative within the same dir, so just recreate it.
            link_target = member.linkname
            # Strip any leading top/ from link target too.
            if link_target.startswith(top + "/"):
                link_target = link_target[len(top) + 1 :]
            target.unlink(missing_ok=True)
            os.symlink(link_target, target)
        elif member.isfile():
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                raise ExtractError(f"Could not extract file {name!r}")
            with src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            target.chmod(member.mode)
        else:
            log.debug("Skipping unsupported tarball member: %s (type=%d)", name, member.type)


# ─────────────────────────────────────────────────────────────────────────────
# Marker file
# ─────────────────────────────────────────────────────────────────────────────


def _marker_content(spec: AssetSpec) -> str:
    return f"{spec.release}\n{spec.sha256}\n{spec.filename}\n"


def _read_marker(path: Path) -> tuple[str, str, str] | None:
    if not path.exists():
        return None
    parts = path.read_text().strip().split("\n")
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


# ─────────────────────────────────────────────────────────────────────────────
# Public install API
# ─────────────────────────────────────────────────────────────────────────────


def is_installed(lib: NativeLib, variant: NativeVariant = NativeVariant.VULKAN) -> bool:
    """True if the pinned release for `lib` is already installed."""
    try:
        spec = resolve_asset(lib, variant)
    except NativeInstallError:
        return False
    marker = lib_install_dir(lib, spec.release) / INSTALLED_MARKER
    found = _read_marker(marker)
    if found is None:
        return False
    release, sha, _fname = found
    if release != spec.release or sha != spec.sha256:
        return False
    return installed_binary_path(lib, spec.release, spec).exists()


def install_status() -> dict[NativeLib, dict[str, str | bool]]:
    """Return a status dict for each lib: release, variant, installed, path."""
    out: dict[NativeLib, dict[str, str | bool]] = {}
    for lib in NativeLib:
        try:
            spec = resolve_asset(lib, NativeVariant.VULKAN)
            installed = is_installed(lib, NativeVariant.VULKAN)
            path = str(installed_binary_path(lib, spec.release, spec)) if installed else ""
            out[lib] = {
                "release": spec.release,
                "variant": NativeVariant.VULKAN.value,
                "installed": installed,
                "path": path,
            }
        except NativeInstallError as exc:
            out[lib] = {
                "release": "",
                "variant": "",
                "installed": False,
                "path": "",
                "error": str(exc),
            }
    return out


def install_lib(
    lib: NativeLib,
    variant: NativeVariant = NativeVariant.VULKAN,
    force: bool = False,
) -> Path:
    """Download, verify, and extract `lib` for the current platform.

    Returns the path to the installed binary. Idempotent: a no-op if the
    pinned release is already installed (unless `force=True`). Concurrent
    installs are serialized via flock.
    """
    spec = resolve_asset(lib, variant)
    install_dir = lib_install_dir(lib, spec.release)
    binary_path = installed_binary_path(lib, spec.release, spec)
    marker_path = install_dir / INSTALLED_MARKER

    if not force and is_installed(lib, variant):
        log.info("%s %s already installed at %s", lib.value, spec.release, install_dir)
        return binary_path

    lock_path = _lock_path()
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)

        # Re-check after acquiring the lock — another process may have installed.
        if not force and is_installed(lib, variant):
            log.info("%s %s installed by another process", lib.value, spec.release)
            return binary_path

        url = _release_url(lib, spec.release, spec.filename)
        log.info("Installing %s %s from %s", lib.value, spec.release, url)

        with tempfile.TemporaryDirectory(prefix=f"dictum-{lib.value}-") as workdir:
            work = Path(workdir)
            archive = work / spec.filename

            log.info("Downloading %s …", spec.filename)
            _download(url, archive)

            log.info("Verifying SHA-256 …")
            actual = _sha256_file(archive)
            if actual != spec.sha256:
                raise HashMismatchError(
                    f"SHA-256 mismatch for {spec.filename}: expected {spec.sha256}, got {actual}"
                )
            log.info("SHA-256 OK")

            log.info("Extracting …")
            extract_dir = work / "extract"
            with tarfile.open(archive, "r:gz") as tf:
                _extract_flat(tf, extract_dir, spec.top_dir)

            # Locate the binary and rename if needed.
            src_bin = extract_dir / spec.binary_name
            if not src_bin.exists():
                raise ExtractError(
                    f"Binary {spec.binary_name!r} not found in tarball (top dir {spec.top_dir!r})"
                )
            if spec.binary_rename:
                src_bin = src_bin.rename(extract_dir / spec.binary_rename)

            # Atomic-ish swap: rmtree old, move new into place, write marker.
            if install_dir.exists():
                shutil.rmtree(install_dir)
            install_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(extract_dir), str(install_dir))

            # Ensure the binary is executable.
            final_bin = install_dir / (spec.binary_rename or spec.binary_name)
            final_bin.chmod(0o755)

            marker_path.write_text(_marker_content(spec))

        log.info("%s %s installed at %s", lib.value, spec.release, install_dir)
        return final_bin


def ensure_installed(lib: NativeLib, variant: NativeVariant = NativeVariant.VULKAN) -> Path:
    """Ensure `lib` is installed; download on first call. Returns binary path.

    Used by the daemon and one-shot CLI on first start. For explicit
    pre-fetching, prefer `dictum native install`.
    """
    return install_lib(lib, variant, force=False)


def installed_binary(lib: NativeLib, variant: NativeVariant = NativeVariant.VULKAN) -> Path | None:
    """Return the path to an already-installed binary, or None if missing.

    Does NOT trigger a download. Useful for status reporting and for
    honoring DICTUM_*_BIN overrides without forcing a fetch.
    """
    try:
        spec = resolve_asset(lib, variant)
    except NativeInstallError:
        return None
    binary = installed_binary_path(lib, spec.release, spec)
    return binary if binary.exists() else None
