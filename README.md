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

- **GPU**: Discrete GPU with Vulkan support (optional but recommended)
- **Vulkan loader**: `libvulkan1` / `vulkan-loader` + a compatible ICD
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

The wheel is pure Python and tiny (tens of KB). The native C++ binaries
are no longer built from source at install time. Instead, Dictum
downloads pinned upstream releases on first use:

- `llama.cpp` → `llama-server` (Vulkan build, `b9699`)
- `CrispASR` → `parakeet-main` (Vulkan build, `v0.7.2`)

Pre-fetch them explicitly (recommended before binding a hotkey):

```bash
dictum native install            # fetch both, Vulkan variant
dictum native install --lib llama
dictum native install --variant cpu   # if no Vulkan ICD is present
dictum native status             # show what's installed and where
```

Binaries land under `$DICTUM_NATIVE_DIR` when set, otherwise under
`$XDG_DATA_HOME/dictum/native/`, or `~/.local/share/dictum/native/`.
The `llama-server` binary ships with `RUNPATH=$ORIGIN` so it finds its
sibling `.so` files automatically; `parakeet-main` is statically linked.
No `LD_LIBRARY_PATH` is required in normal use.

To override the bundled binaries entirely, set `DICTUM_LLM_BIN` and/or
`DICTUM_PARAKEET_BIN` to point at your own builds — Dictum will not
attempt to download when an override is set.

### 3. Download Models

```bash
# Parakeet ASR (multilingual, 4-bit quant)
huggingface-cli download cstr/parakeet-tdt-0.6b-v3-GGUF \
  parakeet-tdt-0.6b-v3-q4_k.gguf \
  --local-dir ~/.cache/dictum/models/

# Qwen LLM (Q3 quant for polishing)
huggingface-cli download unsloth/Qwen3.5-4B-GGUF \
  Qwen3.5-4B-Q3_K_M.gguf \
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
# Install, enable (starts at login), but do not start now
dictum service install

# Install and start immediately
dictum service install --now

# Check status
dictum service status

# Remove the service
dictum service uninstall
```

The unit file is bundled with the wheel and copied to
`~/.config/systemd/user/dictum.service` (or `$XDG_CONFIG_HOME/systemd/user/`).
User units start at graphical/login session, not at boot.

The bundled unit (`dictum/data/dictum.service`):
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

You can still manage the service directly with `systemctl --user
{start,stop,restart,status} dictum`. Edit the installed unit file to
customize `ExecStart` (e.g. add `--profile` flags) and run
`systemctl --user daemon-reload`.

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
