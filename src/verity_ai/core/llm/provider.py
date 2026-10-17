from abc import ABC, abstractmethod
from typing import Any

from fastapi.responses import Response


class AIProvider(ABC):
    name: str
    title: str
    description: str = ""

    @abstractmethod
    def supports(self, model: str) -> bool: ...

    @abstractmethod
    def resolve_model(self, model: str) -> str | None: ...

    @abstractmethod
    def chat(self, data: dict[str, Any]) -> Response | dict[str, Any]: ...

    @abstractmethod
    def aliases(self) -> list[str]: ...

    @abstractmethod
    def models(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def tags(self) -> list[dict[str, str]]: ...
