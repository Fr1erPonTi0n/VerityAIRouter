from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass(frozen=True)
class TranscriptionSegment:
    id: int
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None = None
    language_probability: float | None = None
    duration: float | None = None
    segments: tuple[TranscriptionSegment, ...] = ()


class STTProvider(ABC):
    name: str
    title: str
    description: str = ""

    @abstractmethod
    def models(self) -> list[str]: ...

    @abstractmethod
    def supports(self, model: str) -> bool: ...

    @abstractmethod
    def resolve_model(self, model: str) -> str: ...

    @abstractmethod
    def transcribe(
        self,
        model: str,
        audio: BinaryIO | str | Path,
        language: str | None = None,
    ) -> TranscriptionResult: ...
