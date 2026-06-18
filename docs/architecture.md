# Architecture

## Decision

Dictum will use a persistent daemon with a companion CLI.

The daemon owns the recording lifecycle, state machine, model clients, prompt
profiles, and output dispatch. The CLI sends commands to the daemon over local
IPC and exits quickly, which makes it suitable for Hyprland and Sway hotkeys.

One-shot CLI mode remains supported for scripting and debugging, but the primary
interactive path should go through the daemon.

## Why A Daemon

The target workflow is latency-sensitive. Starting Python, initializing audio
capture, loading ASR weights, and loading an LLM for every hotkey press would
make the dictation loop feel slow and fragile.

A daemon gives us:

- one authoritative state machine;
- warm recorder, ASR, and LLM connections;
- better error reporting for hotkey-triggered operations;
- a stable IPC surface for status widgets or future UI clients;
- clean separation between the hotkey command and long-running work.

## Process Model

```text
+------------------+        local IPC        +-------------------+
| dictum CLI       | ----------------------> | dictum daemon     |
| hotkey friendly  |                         | state + pipeline  |
+------------------+                         +---------+---------+
                                                       |
                                                       v
                              +-----------+    +-------------+    +-----------+
                              | recorder  | -> | ASR backend | -> | LLM       |
                              +-----------+    +-------------+    +-----------+
                                                       |
                                                       v
                                                +--------------+
                                                | output sink  |
                                                +--------------+
```

## State Machine

The daemon exposes these states:

- `idle`: no active job;
- `recording`: microphone capture is active;
- `transcribing`: audio is being converted to text;
- `polishing`: text is being rewritten by an LLM prompt;
- `pasting`: final text is being delivered;
- `failed`: the last operation failed and error details are available.

Expected transitions:

```text
idle -> recording -> transcribing -> polishing -> pasting -> idle
idle -> recording -> transcribing -> pasting -> idle
any active state -> failed -> idle
```

`polishing` may be skipped if the profile disables LLM rewriting.

## CLI Responsibilities

The CLI should:

- parse hotkey-friendly commands and profile overrides;
- send `start`, `stop`, `toggle`, `cancel`, `status`, and `once` requests;
- support output choices: `paste`, `clipboard`, `stdout`, `file`, and `none`;
- return meaningful exit codes;
- avoid owning long-lived model state.

Example commands:

```bash
dictum toggle --profile default --result paste
dictum once --prompt "Make this concise." --result stdout
dictum status --json
```

## Daemon Responsibilities

The daemon should:

- serialize recording jobs unless explicit concurrency is introduced later;
- store state in memory and expose it through IPC;
- load configuration profiles;
- coordinate recorder, ASR, LLM, and output sink backends;
- write short-lived sockets and lock files under `$XDG_RUNTIME_DIR/dictum`;
- optionally run under `systemd --user`.

## IPC

The first implementation should use a Unix domain socket with JSON messages.
This is simple to debug, works well under user sessions, and avoids a hard
dependency on D-Bus. The wire format should be versioned from the start.

Future D-Bus support can be added as another frontend without changing the core
pipeline.

## ASR Backend

The default ASR target is NVIDIA Parakeet TDT v3 0.6B. The backend should be
selected by name and configured with model options, device, precision, and cache
paths.

The core should depend on an interface:

```python
class AsrBackend:
    async def transcribe(self, audio_path: Path) -> Transcript: ...
```

Parakeet runs out-of-process via a pre-built `crispasr-cli` binary (renamed
`parakeet-main`) fetched from the pinned CrispASR GitHub release. Performance
matters more than keeping all logic inside one Python interpreter, so the ASR
implementation may use a native executable, a Python worker with a different
inference framework, or a local service that keeps GPU resources warm. The
daemon should treat ASR as a worker boundary with clear input and output
contracts.

The binary is downloaded on first run by `dictum.native_installer` into
`$XDG_DATA_HOME/dictum/native/crispasr/<release>/` and launched by the daemon
in `--server` mode (HTTP on port 8081) for warm latency. Users can pre-fetch it
with `dictum native install`, or override the path entirely with
`DICTUM_PARAKEET_BIN`.

The first worker protocol can be simple: the daemon writes a recorded audio file
and invokes or requests transcription from the configured ASR worker. Later
optimizations can add streaming audio, persistent worker sockets, or framework
specific fast paths without changing the daemon state machine.

## Recording Backend

The default recorder should use the system's main microphone as defined by
PipeWire. This matches normal desktop behavior and keeps the hotkey workflow
predictable.

The audio source must be configurable by profile and CLI argument for users with
multiple microphones, virtual sources, or special PipeWire routing. The recorder
interface should preserve the distinction between:

- default system source;
- explicit PipeWire source name or node id;
- future non-PipeWire recorder backends.

Dictation is expected to happen in a quiet environment. Use simple speech and
silence detection by default, enough to trim leading and trailing quiet regions
or optionally stop after a silence timeout. Avoid complex diarization or noisy
environment tuning until there is a concrete need.

## LLM Backend

The preferred local LLM path is an OpenAI-compatible HTTP endpoint, such as a
`llama.cpp` server with the chosen Qwen model already loaded on GPU. That keeps
model lifetime outside the daemon and gives predictable warm latency.

The `managed-local` backend ships a pre-built `llama-server` binary (Vulkan
build, pinned to a specific llama.cpp release) downloaded on first run into
`$XDG_DATA_HOME/dictum/native/llama/<release>/`. The binary has
`RUNPATH=$ORIGIN` so it finds its sibling shared libraries without
`LD_LIBRARY_PATH`. Users can pre-fetch via `dictum native install` or override
the path with `DICTUM_LLM_BIN`.

The same interface can support:

- local `llama.cpp` server;
- local vLLM or similar OpenAI-compatible server;
- cloud OpenAI-compatible APIs;
- a no-op backend for raw transcription.

The daemon should not assume that a model name uniquely identifies a provider.
Profiles should separately define provider URL, model, prompt, temperature,
token budget, and timeout.

## Prompt Profiles

Profiles are named configuration blocks. Each profile can define:

- recorder source and silence behavior;
- ASR backend and model settings;
- LLM backend, model, and prompt;
- output target;
- paste strategy;
- post-processing rules.

Prompt text can be supplied inline, from a file, or from a profile.

## Output Backends

Wayland text insertion is environment-dependent, so output should be layered:

1. `wtype` for direct typing when available;
2. `ydotool` for systems where it is configured and permitted;
3. `wl-copy` plus compositor paste binding;
4. `stdout` or file output for scripts.

The selected strategy should be explicit in the command or profile. Auto-detect
can provide a default, but the final choice should be visible in `dictum status`.

Clipboard preservation is the default behavior. When Dictum uses the clipboard
as a paste transport, it should restore the previous clipboard contents after
delivery unless the user opts out. This is still useful with clipboard managers
such as `cliphist`: history can record the transient text, but the active
clipboard should return to its original value for normal desktop use.

Clipboard preservation must be configurable by profile and command-line
arguments because some workflows intentionally want the dictated text to remain
in the clipboard.

## Configuration Layout

Suggested paths:

```text
$XDG_CONFIG_HOME/dictum/config.toml
$XDG_CONFIG_HOME/dictum/prompts/*.md
$XDG_RUNTIME_DIR/dictum/dictum.sock
$XDG_RUNTIME_DIR/dictum/state.json
$XDG_CACHE_HOME/dictum/audio/
```

## Service Management

Support both:

- `dictum daemon` for direct foreground execution;
- `systemd --user` service for normal daily use, installed via
  `dictum service install` (the unit file is bundled with the wheel as
  `dictum/data/dictum.service` and copied to
  `$XDG_CONFIG_HOME/systemd/user/dictum.service`). `dictum service status`
  and `dictum service uninstall` round out the lifecycle.

## Settled Initial Decisions

- Parakeet runs out-of-process through a worker, native executable, or local
  service so transcription performance can be optimized independently.
- Native C++ binaries (llama.cpp, CrispASR) are no longer built from source at
  wheel-build time. Pinned upstream GitHub releases are downloaded on first run
  (or via `dictum native install`) into `$XDG_DATA_HOME/dictum/native/`.
- The default audio input is the system's main PipeWire microphone.
- Audio source selection is configurable by profile and CLI arguments.
- Silence and speech detection should stay simple because the target dictation
  environment is quiet.
- Clipboard contents are restored after paste by default.
- Clipboard preservation is configurable by profile and CLI arguments.
