import importlib
import pkgutil
from functools import lru_cache

from verity_ai.core.tts.voice import VoiceProvider

PACKAGE_NAME = "verity_ai.modules.tts_modules"


@lru_cache(maxsize=1)
def get_voice_providers() -> tuple[VoiceProvider, ...]:
    package = importlib.import_module(PACKAGE_NAME)
    providers: list[VoiceProvider] = []

    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_"):
            continue

        module = importlib.import_module(f"{PACKAGE_NAME}.{module_info.name}")
        voice_provider = getattr(module, "voice_provider", None)

        if isinstance(voice_provider, VoiceProvider):
            providers.append(voice_provider)

    return tuple(providers)


def find_voice(voice: str) -> tuple[VoiceProvider, str] | None:
    for provider in get_voice_providers():
        if voice in provider.voices():
            return provider, voice

    return None


def available_voices() -> list[str]:
    return sorted(
        {
            voice
            for provider in get_voice_providers()
            for voice in provider.voices()
        }
    )
