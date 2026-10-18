import importlib
import pkgutil
from functools import lru_cache

from verity_ai.core.stt.provider import STTProvider

PACKAGE_NAME = "verity_ai.modules.stt_modules"


@lru_cache(maxsize=1)
def get_stt_providers() -> tuple[STTProvider, ...]:
    package = importlib.import_module(PACKAGE_NAME)
    providers: list[STTProvider] = []

    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_"):
            continue

        module = importlib.import_module(f"{PACKAGE_NAME}.{module_info.name}")
        stt_provider = getattr(module, "stt_provider", None)

        if isinstance(stt_provider, STTProvider):
            providers.append(stt_provider)

    return tuple(providers)


def find_stt_provider(model: str) -> STTProvider | None:
    return next(
        (provider for provider in get_stt_providers() if provider.supports(model)),
        None,
    )


def available_stt_models() -> list[str]:
    return sorted(
        {
            model
            for provider in get_stt_providers()
            for model in provider.models()
        }
    )
