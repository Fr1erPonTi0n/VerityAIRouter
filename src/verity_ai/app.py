from fastapi import FastAPI

from verity_ai.registry import AIModule, discover_modules

MODULES: list[AIModule] = discover_modules()

app = FastAPI(title="Verity AI Router", version="0.1.0")

for module in MODULES:
    if module.router is not None:
        app.include_router(module.router)


@app.get("/api/ai/modules")
async def get_modules():
    return {
        "modules": [
            {
                "name": module.name,
                "title": module.title,
                "description": module.description,
                "kind": module.kind,
                "router": module.router is not None,
                "provider": module.provider is not None,
                "stt_provider": module.stt_provider is not None,
                "voice_provider": module.voice_provider is not None,
            }
            for module in MODULES
        ]
    }
