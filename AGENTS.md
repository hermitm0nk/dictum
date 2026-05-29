# AGENTS.md

Guidance for agents working on this repository.

## Project Intent

Dictum is a Python CLI plus daemon for low-latency dictation on Wayland. Keep
the hotkey path fast, predictable, and scriptable. The daemon should own
long-lived state and expensive model resources; the CLI should remain a small
client suitable for Hyprland or Sway bindings.

## Engineering Preferences

- Prefer typed Python and small interfaces over framework-heavy code.
- Keep platform-specific code behind backend classes.
- Keep model integrations pluggable. Local GPU inference and cloud APIs should
  share the same high-level interface.
- Do not block the CLI on long work except for explicit one-shot commands.
- Treat Wayland output tools as external dependencies and detect them clearly.
- Store runtime state under `$XDG_RUNTIME_DIR/dictum` when available.
- Store user configuration under `$XDG_CONFIG_HOME/dictum`.

## Commands

```bash
python -m dictum --help
pytest
ruff check .
ruff format .
mypy src
```

## Safety

Do not commit model weights, recordings, transcripts, API keys, or generated
private text. Use `.gitignore` for local artifacts.
