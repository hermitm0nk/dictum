# Dictum

Dictum is a Python voice-to-text command tool designed for Wayland compositors
such as Hyprland and Sway. It is intended to be triggered from a hotkey,
record microphone audio, transcribe it with NVIDIA Parakeet TDT v3 0.6B, polish
the transcription with a local or cloud LLM, and send the final text to the
active window, clipboard, stdout, or a file.

This repository is currently a project scaffold and architecture baseline.

## Goals

- Fast hotkey workflow for dictation and text transformation.
- Persistent daemon for state, warm models, and lower latency.
- Companion CLI for key bindings and scripting.
- Pluggable ASR, LLM, recorder, and output backends.
- Wayland-first output through `wtype`, `ydotool`, `wl-copy`, or stdout.

## Initial Workflow

```text
Hyprland/Sway hotkey
  -> dictum CLI
  -> daemon command over local IPC
  -> record mic audio
  -> transcribe with Parakeet
  -> polish with LLM prompt
  -> paste/copy/print/save result
```

## Example Usage

```bash
# Toggle recording from a window-manager binding.
dictum toggle --profile default

# Start and stop explicitly.
dictum start --profile emails
dictum stop --result paste

# One-shot command with explicit backends.
dictum once \
  --asr parakeet-tdt-v3-0.6b \
  --llm http://127.0.0.1:8080/v1/chat/completions \
  --prompt-file prompts/polish.md \
  --result clipboard

# Inspect daemon state.
dictum status
```

## Suggested Hyprland Binding

```ini
bind = SUPER, D, exec, dictum toggle --profile default --result paste
```

## Development

This project targets Python 3.11+.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
dictum --help
```

## Documentation

- [Architecture](docs/architecture.md)
- [Development Notes](AGENTS.md)

## Status Model

Dictum exposes a simple state machine:

- `idle`
- `recording`
- `transcribing`
- `polishing`
- `pasting`
- `failed`

The daemon owns the authoritative state. The CLI is a thin control surface that
can be safely called from hotkeys.
