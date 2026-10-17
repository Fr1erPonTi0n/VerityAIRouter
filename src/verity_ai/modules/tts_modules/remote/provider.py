from verity_ai.config import (
    TTS_API_KEY,
    TTS_API_TIMEOUT,
    TTS_API_TRUST_ENV,
    TTS_API_URL,
    TTS_API_VOICE,
    TTS_API_VOICES,
)
from verity_ai.core.tts.api import APIVoiceProvider, TTSAPIError


class RemoteTTSProvider(APIVoiceProvider):
    name = "remote"
    title = "Remote TTS"
    description = "Удалённый TTS-сервер по HTTP API"

    def __init__(self) -> None:
        self.url = TTS_API_URL
        self.api_key = TTS_API_KEY
        self.timeout = TTS_API_TIMEOUT
        self.trust_env = TTS_API_TRUST_ENV

    def voices(self) -> list[str]:
        voices = [
            voice.strip()
            for voice in TTS_API_VOICES.split(",")
            if voice.strip()
        ]

        return voices or [TTS_API_VOICE]

    def synthesize(self, voice: str, text: str) -> bytes:
        response = self._post_json(
            self.url,
            {
                "input": text,
                "voice": voice,
                "response_format": "wav",
            },
        )

        if not response.content:
            raise TTSAPIError("Удалённый TTS-сервер вернул пустой ответ.")

        return response.content
