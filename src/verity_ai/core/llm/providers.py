import importlib
import pkgutil
from functools import lru_cache

from verity_ai.core.llm.provider import AIProvider

PACKAGE_NAME = "verity_ai.modules.llm_modules"


@lru_cache(maxsize=1)
def get_providers() -> tuple[AIProvider, ...]:
    package = importlib.import_module(PACKAGE_NAME)
    providers: list[AIProvider] = []

    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_"):
            continue

        module = importlib.import_module(f"{PACKAGE_NAME}.{module_info.name}")
        provider = getattr(module, "provider", None)

        if isinstance(provider, AIProvider):
            providers.append(provider)

    return tuple(providers)


def find_provider(model: str) -> AIProvider | None:
    return next(
        (provider for provider in get_providers() if provider.supports(model)),
        None,
    )


def available_models() -> list[str]:
    return sorted(
        {
            model
            for provider in get_providers()
            for model in provider.aliases()
        }
    )
