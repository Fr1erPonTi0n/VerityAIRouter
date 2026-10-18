import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]

load_dotenv(ROOT_DIR / ".env")


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_path(name: str, default: Path) -> Path:
    path = Path(_env(name, str(default))).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


HOST = _env("HOST", "127.0.0.1")
PORT = int(_env("PORT", "8880"))

DEEPSEEK_API_KEY = _env("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = _env("DEEPSEEK_URL", "https://api.deepseek.com/chat/completions")
OLLAMA_URL = _env("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
REQUEST_TIMEOUT = float(_env("REQUEST_TIMEOUT", "300"))

DEFAULT_MODEL = _env("DEFAULT_MODEL", "deepseek-flash")
DEFAULT_TTS_VOICE = _env("DEFAULT_TTS_VOICE", "ruslan")

STT_MODEL = _env("STT_MODEL", "small")
STT_DEVICE = _env("STT_DEVICE", "auto")
STT_COMPUTE_TYPE = _env("STT_COMPUTE_TYPE", "int8")
STT_LANGUAGE = _env("STT_LANGUAGE", "")
STT_BEAM_SIZE = int(_env("STT_BEAM_SIZE", "5"))

TTS_API_URL = _env("TTS_API_URL", "")
TTS_API_KEY = _env("TTS_API_KEY", "")
TTS_API_TIMEOUT = float(_env("TTS_API_TIMEOUT", "30"))
TTS_API_VOICE = _env("TTS_API_VOICE", "default")
TTS_API_VOICES = _env("TTS_API_VOICES", "")
TTS_API_TRUST_ENV = _env("TTS_API_TRUST_ENV", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

TTS_MODEL_DIR = _env_path(
    "TTS_MODEL_DIR",
    ROOT_DIR / "src" / "verity_ai" / "data" / "tts",
)
LLM_MODEL_DIR = _env_path(
    "LLM_MODEL_DIR",
    ROOT_DIR / "src" / "verity_ai" / "data" / "llm",
)
STT_MODEL_DIR = _env_path(
    "STT_MODEL_DIR",
    ROOT_DIR / "src" / "verity_ai" / "data" / "stt",
)
