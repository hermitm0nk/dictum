from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from dictum.models import Profile, ResultTarget
from dictum.state import DictumState

app = typer.Typer(no_args_is_help=True, help="Wayland hotkey dictation CLI.")
console = Console()


def _profile_from_args(
    profile: str,
    result: ResultTarget,
    asr_model: str | None,
    llm_model: str | None,
    prompt: str | None,
    prompt_file: Path | None,
) -> Profile:
    loaded = Profile(name=profile, result=result)
    if asr_model:
        loaded.asr.model = asr_model
    if loaded.llm and llm_model:
        loaded.llm.model = llm_model
    if prompt:
        loaded.prompt = prompt
    if prompt_file:
        loaded.prompt_file = prompt_file
    return loaded


@app.command()
def daemon() -> None:
    """Run the daemon in the foreground."""
    console.print("dictum daemon scaffold: IPC server not implemented yet")


@app.command()
def status(json: Annotated[bool, typer.Option("--json", help="Emit JSON status.")] = False) -> None:
    """Print daemon state."""
    state = {"state": DictumState.IDLE.value, "active_profile": None, "last_error": None}
    if json:
        console.print_json(data=state)
    else:
        console.print(f"state: {state['state']}")


@app.command()
def start(
    profile: Annotated[str, typer.Option(help="Profile name.")] = "default",
) -> None:
    """Start recording through the daemon."""
    console.print(f"start requested for profile: {profile}")


@app.command()
def stop(
    result: Annotated[
        ResultTarget, typer.Option(help="Where to send the final text.")
    ] = ResultTarget.PASTE,
) -> None:
    """Stop recording and process the captured audio."""
    console.print(f"stop requested with result target: {result}")


@app.command()
def toggle(
    profile: Annotated[str, typer.Option(help="Profile name.")] = "default",
    result: Annotated[
        ResultTarget, typer.Option(help="Where to send the final text.")
    ] = ResultTarget.PASTE,
) -> None:
    """Toggle recording state through the daemon."""
    console.print(f"toggle requested for profile: {profile}, result: {result}")


@app.command()
def once(
    profile: Annotated[str, typer.Option(help="Profile name.")] = "default",
    result: Annotated[
        ResultTarget, typer.Option(help="Where to send the final text.")
    ] = ResultTarget.STDOUT,
    asr_model: Annotated[str | None, typer.Option("--asr", help="ASR model name.")] = None,
    llm_model: Annotated[str | None, typer.Option("--llm", help="LLM model name or alias.")] = None,
    prompt: Annotated[str | None, typer.Option(help="Inline polishing instruction.")] = None,
    prompt_file: Annotated[Path | None, typer.Option(help="Path to polishing prompt.")] = None,
) -> None:
    """Run one dictation job in the current process."""
    loaded = _profile_from_args(profile, result, asr_model, llm_model, prompt, prompt_file)
    console.print(f"one-shot scaffold for profile: {loaded.name}")
    console.print(f"asr: {loaded.asr.model}")
    console.print(f"llm: {loaded.llm.model if loaded.llm else 'disabled'}")
    console.print(f"result: {loaded.result}")


if __name__ == "__main__":
    app()
