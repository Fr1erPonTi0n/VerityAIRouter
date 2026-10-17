from verity_ai.config import TTS_API_URL
from verity_ai.modules.tts_modules.remote.provider import RemoteTTSProvider

voice_provider = RemoteTTSProvider() if TTS_API_URL else None

NAME = "remote"
TITLE = "Remote TTS"
DESCRIPTION = "Удалённый TTS-сервер по HTTP API"
KIND = "tts"
