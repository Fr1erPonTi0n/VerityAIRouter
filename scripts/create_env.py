#!/usr/bin/env python3
import getpass
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"

FIELDS = {
    "HOST": "127.0.0.1",
    "PORT": "8880",
    "DEEPSEEK_API_KEY": "",
    "DEEPSEEK_URL": "https://api.deepseek.com/chat/completions",
    "OLLAMA_URL": "http://127.0.0.1:11434/api/chat",
    "REQUEST_TIMEOUT": "300",
    "DEFAULT_MODEL": "deepseek-flash",
    "DEFAULT_TTS_VOICE": "irina",
    "TTS_MODEL_DIR": "src/verity_ai/data/tts",
    "LLM_MODEL_DIR": "src/verity_ai/data/llm",
    "TTS_API_URL": "",
    "TTS_API_KEY": "",
    "TTS_API_TIMEOUT": "30",
    "TTS_API_VOICE": "default",
    "TTS_API_VOICES": "",
    "TTS_API_TRUST_ENV": "false",
}

SECTIONS = (
    ("Сервер", ("HOST", "PORT")),
    ("DeepSeek", ("DEEPSEEK_API_KEY", "DEEPSEEK_URL")),
    ("Ollama", ("OLLAMA_URL",)),
    ("HTTP-запросы", ("REQUEST_TIMEOUT",)),
    ("Модели по умолчанию", ("DEFAULT_MODEL", "DEFAULT_TTS_VOICE")),
    ("Каталоги моделей", ("TTS_MODEL_DIR", "LLM_MODEL_DIR")),
    (
        "Удалённый TTS",
        (
            "TTS_API_URL",
            "TTS_API_KEY",
            "TTS_API_TIMEOUT",
            "TTS_API_VOICE",
            "TTS_API_VOICES",
            "TTS_API_TRUST_ENV",
        ),
    ),
)

SECRET_FIELDS = {"DEEPSEEK_API_KEY", "TTS_API_KEY"}


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}

    if not path.exists():
        return values

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")

    return values


def ask(name: str, default: str, secret: bool) -> str:
    if not sys.stdin.isatty():
        env_value = os.getenv(name)
        return env_value if env_value is not None else default

    shown = "********" if secret and default else default
    prompt = f"{name} [{shown}]: " if shown else f"{name}: "

    if secret:
        value = getpass.getpass(prompt)
    else:
        value = input(prompt).strip()

    return value or default


def build_content(values: dict[str, str], extras: dict[str, str]) -> str:
    lines = [
        "# Verity AI Router",
        "# Создано scripts/create_env.py",
        "",
    ]

    for title, names in SECTIONS:
        lines.append(f"# {title}")
        lines.extend(f"{name}={values[name]}" for name in names)
        lines.append("")

    if extras:
        lines.append("# Дополнительные переменные")
        lines.extend(f"{name}={value}" for name, value in extras.items())
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    force = "--force" in sys.argv[1:]

    if ENV_PATH.exists() and not force:
        print(f"{ENV_PATH} уже существует. Используйте --force для перезаписи.")
        return 1

    existing = read_env(ENV_PATH)
    values: dict[str, str] = {}

    for name, default in FIELDS.items():
        values[name] = ask(
            name,
            existing.get(name, default),
            name in SECRET_FIELDS,
        )

    extras = {
        name: value
        for name, value in existing.items()
        if name not in values
    }

    ENV_PATH.write_text(build_content(values, extras), encoding="utf-8")

    print(f"Создан {ENV_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
