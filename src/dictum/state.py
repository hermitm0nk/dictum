from enum import StrEnum


class DictumState(StrEnum):
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    POLISHING = "polishing"
    PASTING = "pasting"
    FAILED = "failed"
