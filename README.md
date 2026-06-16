# Dictum

Dictum is a Python voice-to-text command tool designed for Wayland compositors
such as Hyprland and Sway. It is intended to be triggered from a hotkey,
record microphone audio, transcribe it with Parakeet TDT v3 0.6B, polish
the transcription with a local or cloud LLM, and send the final text to the
active window, clipboard, stdout, or a file.

## Goals

- Fast hotkey workflow for dictation and text transformation.
- Persistent daemon for state, warm models, and lower latency.
- Companion CLI for key bindings and scripting.
- Pluggable ASR, LLM, recorder, and output backends.
- Wayland-first output through `wtype`, `ydotool`, `wl-copy`, or stdout.

## Architecture

```
Hyprland/Sway hotkey
  -> dictum CLI (thin IPC client)
  -> daemon command over local Unix socket
  -> record mic audio (sounddevice/PipeWire)
  -> transcribe with Parakeet (CrispASR HTTP server, port 8081)
  -> polish with LLM (llama.cpp server, port 8080)
  -> paste/copy/print/save result
```

### Moving Parts

| Component | Technology | Purpose |
|-----------|------------|---------|
| **CLI** | Python + Typer | Hotkey-friendly commands (`toggle`, `start`, `stop`, `once`, `status`) |
| **Daemon** | Python + asyncio | State machine, IPC server, pipeline orchestration |
| **Recorder** | sounddevice (PortAudio) | PipeWire/PulseAudio capture, silence watchdog |
| **ASR** | CrispASR (C++/Vulkan) | Parakeet TDT v3 0.6B GGUF, HTTP server mode (warm) |
| **LLM** | llama.cpp (C++/Vulkan) | Qwen 3.5-4B GGUF, OpenAI-compatible HTTP server (warm) |
| **Output** | wtype / ydotool / wl-copy | Wayland text insertion |
| **Notify** | D-Bus (gdbus) | Desktop notifications |

### State Machine

```
idle -> recording -> transcribing -> polishing -> pasting -> idle
              \-> failed (on error) ----------------------------> idle
```

`polishing` is skipped if the profile disables LLM rewriting.

---

## Quick Start

### 1. Prerequisites

- **GPU**: Discrete GPU with Vulkan support
- **Vulkan SDK**: `vulkan-devel` (Arch) / `vulkan-sdk` (others)
- **Build tools**: `cmake`, `ninja`, `gcc`/`clang`
- **Python**: 3.11+ with `uv` (recommended) or `pip`

### 2. Install Dictum

```bash
# From PyPI, once published
uv tool install dictum

# Or from a local source checkout while developing
git clone https://github.com/hermitm0nk/dictum.git
cd dictum
uv tool install .
```

**What gets built:**
- `llama.cpp` → `llama-server` (Vulkan backend, no CUDA)
- `CrispASR` → `parakeet-main` (Parakeet TDT v0.6b GGUF)

The wheel contains both binaries and their shared libraries under
`dictum/_native/`. Both binaries are built with
`CMAKE_INSTALL_RPATH='$ORIGIN/../lib/<name>'` so they find their bundled shared
libraries without `LD_LIBRARY_PATH`.

Native dependency source checkouts are cached under
`$DICTUM_NATIVE_SOURCE_CACHE` when set, otherwise under
`$XDG_CACHE_HOME/dictum/native-src` or `~/.cache/dictum/native-src`. CMake build
directories stay temporary so repeated local builds reuse git checkouts without
reusing stale build-tool paths from isolated Python build environments.

### 3. Download Models

```bash
# Parakeet ASR (multilingual, 4-bit quant)
huggingface-cli download cstr/parakeet-tdt-0.6b-v3-GGUF \
  parakeet-tdt-0.6b-v3-q4_k.gguf \
  --local-dir ~/.cache/dictum/models/

# Qwen LLM (3-bit quant for polishing)
huggingface-cli download Qwen/Qwen2.5-4B-Instruct-GGUF \
  qwen2.5-4b-instruct-q3_k_m.gguf \
  --local-dir ~/.cache/dictum/models/
```

By default Dictum downloads models on first use into
`$DICTUM_MODEL_DIR` when set, otherwise into
`$XDG_CACHE_HOME/dictum/models` or `~/.cache/dictum/models`.

### 4. Configure (Optional)

```bash
mkdir -p ~/.config/dictum
cp examples/config.toml ~/.config/dictum/config.toml
# Edit profiles, ASR/LLM settings, output target
```

---

## Usage

### Start the Daemon

```bash
# Foreground (testing)
dictum daemon --profile default

# Or as systemd user service (persistent, see below)
```

### Hotkey Commands

```bash
# Toggle recording (press once to start, again to stop & process)
dictum toggle --profile default --result paste

# Explicit start/stop
dictum start --profile default
dictum stop --result paste

# One-shot (no daemon needed)
dictum once --result stdout --duration 30

# Status
dictum status
dictum status --json
```

### Output Targets (`--result`)

| Target | Behavior |
|--------|----------|
| `paste` | Types into focused window via `wtype`/`ydotool` (fallback: clipboard) |
| `clipboard` | Copies to clipboard via `wl-copy` |
| `stdout` | Prints to terminal |
| `file` | Writes to `/tmp/dictum-output.txt` |
| `none` | No output |

---

## Systemd User Service (Persistent Daemon)

```bash
# Install service (does NOT auto-start on boot)
mkdir -p ~/.config/systemd/user
cp packaging/systemd/dictum.service ~/.config/systemd/user/

systemctl --user daemon-reload
systemctl --user enable --now dictum   # start now
# systemctl --user start dictum        # start manually
# systemctl --user stop dictum         # stop manually
```

**Service file** (`packaging/systemd/dictum.service`):
```ini
[Unit]
Description=Dictum voice dictation daemon
After=graphical-session.target pipewire.service

[Service]
Type=simple
ExecStart=%h/.local/bin/dictum daemon
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
```

> **Note**: `enable --now` starts it immediately. Remove `--now` if you prefer manual start.

---

## Hyprland Binding

Add to `~/.config/hypr/hyprland.conf`:

```ini
# Dictum voice dictation (ScrollLock to toggle)
bind = , Scroll_Lock, exec, dictum toggle --profile default --result paste
```

Reload config:
```bash
hyprctl reload
```
