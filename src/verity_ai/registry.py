import importlib
import pkgutil
from dataclasses import dataclass

from fastapi import APIRouter

from verity_ai.core.llm.provider import AIProvider
from verity_ai.core.tts.voice import VoiceProvider

MODULE_PACKAGES = (
    "verity_ai.modules.llm_modules",
    "verity_ai.modules.tts_modules",
)


@dataclass(frozen=True)
class AIModule:
    name: str
    title: str
    description: str
    kind: str
    router: APIRouter | None
    provider: AIProvider | None
    voice_provider: VoiceProvider | None


def iter_module_names():
    for package_name in MODULE_PACKAGES:
        package = importlib.import_module(package_name)

        for module_info in pkgutil.iter_modules(package.__path__):
            if module_info.name.startswith("_"):
                continue

            yield f"{package_name}.{module_info.name}"


def discover_modules() -> list[AIModule]:
    discovered: list[AIModule] = []

    for module_name in iter_module_names():
        module = importlib.import_module(module_name)
        router = getattr(module, "router", None)
        provider = getattr(module, "provider", None)
        voice_provider = getattr(module, "voice_provider", None)

        if (
            not isinstance(router, APIRouter)
            and not isinstance(provider, AIProvider)
            and not isinstance(voice_provider, VoiceProvider)
        ):
            continue

        discovered.append(
            AIModule(
                name=getattr(module, "NAME", module_name.rsplit(".", 1)[-1]),
                title=getattr(module, "TITLE", module_name.rsplit(".", 1)[-1]),
                description=getattr(module, "DESCRIPTION", ""),
                kind=getattr(module, "KIND", "other"),
                router=router if isinstance(router, APIRouter) else None,
                provider=provider if isinstance(provider, AIProvider) else None,
                voice_provider=(
                    voice_provider
                    if isinstance(voice_provider, VoiceProvider)
                    else None
                ),
            )
        )

    return discovered
