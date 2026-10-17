from typing import Any

import requests
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from verity_ai.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_URL,
    DEFAULT_MODEL,
    REQUEST_TIMEOUT,
)
from verity_ai.core.llm import AIProvider


class DeepSeekProvider(AIProvider):
    name = "deepseek"
    title = "DeepSeek"
    description = "Облачный провайдер языковых моделей DeepSeek"

    MODELS = {
        "deepseek": "deepseek-flash",
        "deepseek-chat": "deepseek-flash",
        "deepseek-coder": "deepseek-flash",
        "deepseek-flash": "deepseek-flash",
    }

    def supports(self, model: str) -> bool:
        return model in self.MODELS

    def resolve_model(self, model: str) -> str | None:
        return self.MODELS.get(model)

    def aliases(self) -> list[str]:
        return list(self.MODELS)

    def models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "deepseek-flash",
                "object": "model",
                "created": 1677610602,
                "owned_by": self.name,
            }
        ]

    def tags(self) -> list[dict[str, str]]:
        return []

    def chat(self, data: dict[str, Any]) -> StreamingResponse | dict[str, Any]:
        model = str(data.get("model", DEFAULT_MODEL)).lower().strip()
        deepseek_model = self.resolve_model(model)

        if not deepseek_model:
            raise HTTPException(
                status_code=400,
                detail=f"Неизвестная модель DeepSeek '{model}'.",
            )

        if not DEEPSEEK_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="DEEPSEEK_API_KEY не задан в .env.",
            )

        stream = data.get("stream", False)
        payload = {
            "model": deepseek_model,
            "messages": data.get("messages", []),
            "stream": stream,
            "temperature": data.get("temperature", 0.7),
        }

        for key in (
            "max_tokens",
            "top_p",
            "stop",
            "presence_penalty",
            "frequency_penalty",
        ):
            if key in data:
                payload[key] = data[key]

        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                DEEPSEEK_URL,
                json=payload,
                headers=headers,
                stream=stream,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            status = exc.response.status_code if exc.response is not None else 502
            detail = exc.response.text if exc.response is not None else str(exc)
            raise HTTPException(status_code=status, detail=detail)

        if stream:
            return StreamingResponse(
                response.iter_lines(),
                media_type="text/event-stream",
            )

        return response.json()
