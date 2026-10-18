from fastapi import APIRouter, HTTPException, Request

from verity_ai.config import DEFAULT_MODEL
from verity_ai.core.llm.providers import (
    available_models,
    find_provider,
    get_providers,
)
from verity_ai.core.stt.providers import available_stt_models

router = APIRouter()


@router.post("/api/chat")
@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def proxy_chat(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    model = str(data.get("model", DEFAULT_MODEL)).lower().strip()
    provider = find_provider(model)

    if provider is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Неизвестная модель '{model}'. Доступные: "
                f"{', '.join(available_models())}."
            ),
        )

    return provider.chat(data)


@router.get("/api/tags")
async def get_ollama_models():
    return {
        "models": [
            tag for provider in get_providers() for tag in provider.tags()
        ]
    }


@router.get("/v1/models")
async def get_models():
    data = [
        model for provider in get_providers() for model in provider.models()
    ]
    data.extend(
        {"id": model, "object": "model", "owned_by": "verity"}
        for model in available_stt_models()
    )

    return {"object": "list", "data": data}
