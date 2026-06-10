# Dictum

Dictum is a Python voice-to-text command tool designed for Wayland compositors
such as Hyprland and Sway. It is intended to be triggered from a hotkey,
record microphone audio, transcribe it with NVIDIA Parakeet TDT v3 0.6B, polish
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

- **GPU**: NVIDIA GPU with Vulkan support (see [Why NVIDIA?](#why-nvidia))
- **Vulkan SDK**: `vulkan-devel` (Arch) / `vulkan-sdk` (others)
- **Build tools**: `cmake`, `ninja`, `gcc`/`clang`
- **Python**: 3.11+ with `uv` (recommended) or `pip`

### 2. Install Native Dependencies

```bash
# Clone repo
git clone https://github.com/your-repo/dictum.git
cd dictum

# Build llama.cpp (Vulkan) + CrispASR from source
./scripts/build.sh

# Deploy binaries + libs to ~/.dictum/ (rpath-linked, no LD_LIBRARY_PATH needed)
./scripts/deploy.sh
```

**What gets built:**
- `llama.cpp` → `llama-server` (Vulkan backend, no CUDA)
- `CrispASR` → `parakeet-main` (Parakeet TDT v0.6b GGUF)

Both binaries are built with `CMAKE_INSTALL_RPATH='$ORIGIN/../lib/<name>'` so they
find their shared libraries without `LD_LIBRARY_PATH`.

### 3. Download Models

```bash
# Parakeet ASR (multilingual, 4-bit quant)
huggingface-cli download cstr/parakeet-tdt-0.6b-v3-GGUF \
  parakeet-tdt-0.6b-v3-q4_k.gguf \
  --local-dir ~/.cache/dictum/models/

# Qwen LLM (3-bit quant for polishing)
huggingface-cli download <qwen-repo> <qwen-model>.gguf \
  --local-dir ~/.cache/dictum/models/
```

### 4. Install Python Package

```bash
# As a uv tool (recommended — global `dictum` command, editable)
uv tool install -e .

# Or install globally (non-editable)
uv tool install .

# Or with pipx
pipx install -e .
```

### 5. Configure (Optional)

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

---

## Why NVIDIA GPU? {#why-nvidia}

Dictum uses **Vulkan-backed llama.cpp and CrispASR** for local inference. The choice of NVIDIA over iGPU (AMD Radeon/Ryzen) is driven by:

### 1. **Vulkan Driver Maturity**
- **NVIDIA**: Proprietary driver has production-grade Vulkan support with full feature parity.
- **AMD (RADV)**: Open-source RADV driver is excellent for graphics but historically has gaps in compute workloads (subgroup operations, timeline semaphores, sparse bindings) that llama.cpp/ggml relies on.

### 2. **ggml/llama.cpp Vulkan Backend**
- The ggml Vulkan backend (`ggml-vulkan`) is developed and tested primarily on NVIDIA.
- AMD support exists but hits edge cases: validation errors, missing extensions, performance regressions.
- NVIDIA "just works" for the tensor operations (GEMM, attention, RoPE) used by transformer models.

### 3. **VRAM vs. Shared Memory**
- Discrete NVIDIA GPUs have dedicated VRAM (8–24 GB) → larger models fit entirely on GPU.
- iGPU shares system RAM → limited by bandwidth (DDR5 ~100 GB/s vs. GDDR6 ~500 GB/s), smaller usable allocation.

### 4. **Model Quantization Targets**
- Dictum defaults to **3-bit (Qwen) / 4-bit (Parakeet)** GGUF quantizations.
- These are optimized for GPU tensor cores; CPU fallback is ~10–20× slower.
- On iGPU, the memory bandwidth bottleneck makes quantized inference only marginally faster than CPU.

### 5. **CrispASR / Parakeet**
- CrispASR's GGML backend inherits the same Vulkan constraints.
- Parakeet TDT 0.6B runs acceptably on CPU (single-threaded ~1.5× realtime), but the daemon keeps it warm on GPU for sub-second latency.

### Can It Work on AMD?

Yes, with caveats:
- Set `GGML_VULKAN=1` + `RADV_PERFTEST=ngg,nggc` + `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json`
- May need `LLAMA_VULKAN=1` and latest mesa-git/radv-git
- Expect occasional validation layers errors, lower throughput
- CPU fallback (`-ngl 0`) is always an option — Parakeet on CPU is usable; Qwen 4B on CPU is slow (~2–3 tokens/s)

### Summary

| Factor | NVIDIA (Discrete) | AMD iGPU (Ryzen) |
|--------|-------------------|------------------|
| Vulkan compute | ✅ Mature | ⚠️ Gaps in compute |
| VRAM | 8–24 GB dedicated | Shared system RAM |
| Bandwidth | ~500 GB/s (GDDR6) | ~100 GB/s (DDR5) |
| llama.cpp Vulkan | Primary target | Secondary |
| CrispASR Vulkan | Tested | Limited testing |
| Quantized inference | Fast (tensor cores) | Bandwidth-bound |

**Default configuration assumes NVIDIA + Vulkan**. To use CPU-only: set `n_gpu_layers = 0` in profile or pass `-ngl 0` to llama-server.
