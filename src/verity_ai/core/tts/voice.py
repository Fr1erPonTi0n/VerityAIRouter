from abc import ABC, abstractmethod


class VoiceProvider(ABC):
    name: str
    title: str
    description: str = ""

    @abstractmethod
    def voices(self) -> list[str]: ...

    @abstractmethod
    def synthesize(self, voice: str, text: str) -> bytes: ...
