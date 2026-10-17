from typing import Any

import requests
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from verity_ai.config import (
    DEFAULT_MODEL, 
    OLLAMA_URL, 
    REQUEST_TIMEOUT
)
from verity_ai.core.llm import AIProvider


class OllamaProvider(AIProvider):
    name = "ollama"
    title = "Ollama"
    description = "Локальный провайдер языковых моделей Ollama"

    MODELS = {
        "qwen1": "qwen2.5:3b",
        "qwen2": "qwen2.5:7b",
        "qwen2.5:3b": "qwen2.5:3b",
        "llama": "llama3.2",
        "llama3.2": "llama3.2",
    }

    def supports(self, model: str) -> bool:
        return self.resolve_model(model) is not None

    def resolve_model(self, model: str) -> str | None:
        if model.startswith("ollama-"):
            model = model.removeprefix("ollama-")

        return self.MODELS.get(model)

    def aliases(self) -> list[str]:
        return list(self.MODELS)

    def models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "qwen",
                "object": "model",
                "created": 1677610602,
                "owned_by": self.name,
            },
            {
                "id": "llama",
                "object": "model",
                "created": 1677610602,
                "owned_by": self.name,
            },
        ]

    def tags(self) -> list[dict[str, str]]:
        return [
            {"name": "qwen1", "model": "qwen2.5:3b"},
            {"name": "qwen2", "model": "qwen2.5:7b"},
            {"name": "llama", "model": "llama3.2"},
        ]

    def chat(self, data: dict[str, Any]) -> StreamingResponse | dict[str, Any]:
        model = str(data.get("model", DEFAULT_MODEL)).lower().strip()
        ollama_model = self.resolve_model(model)

        if not ollama_model:
            raise HTTPException(
                status_code=400,
                detail=f"Неизвестная модель Ollama '{model}'.",
            )

        stream = data.get("stream", False)
        payload = {
            "model": ollama_model,
            "messages": data.get("messages", []),
            "stream": stream,
            "options": {
                "temperature": data.get("temperature", 0.7),
            },
        }

        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                stream=stream,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            status = exc.response.status_code if exc.response is not None else 502
            detail = exc.response.text if exc.response is not None else str(exc)
            raise HTTPException(
                status_code=status,
                detail=f"Ollama недоступна: {detail}",
            )

        if stream:
            return StreamingResponse(
                response.iter_lines(),
                media_type="application/x-ndjson",
            )

        result = response.json()
        return {
            "id": "ollama-chat",
            "object": "chat.completion",
            "model": ollama_model,
            "choices": [
                {
                    "index": 0,
                    "message": result.get("message", {}),
                    "finish_reason": "stop",
                }
            ],
        }
