import io
import wave
from pathlib import Path

from piper import PiperVoice

from verity_ai.config import TTS_MODEL_DIR
from verity_ai.core.tts import VoiceProvider


class PiperProvider(VoiceProvider):
    name = "piper"
    title = "Piper"
    description = "Локальные голосовые модели Piper"

    def __init__(self, model_dir: Path = TTS_MODEL_DIR):
        self.model_dir = model_dir
        self._voices: dict[str, PiperVoice] = {}
        self._load_voices()

    def _load_voices(self) -> None:
        print(f"Загрузка моделей Piper TTS из: {self.model_dir}")

        for model_path in self.model_dir.glob("*.onnx"):
            voice_name = model_path.stem
            config_path = model_path.with_suffix(".json")

            if not config_path.exists():
                config_path = Path(f"{model_path}.json")

            if not config_path.exists():
                print(
                    f"Пропуск {model_path.name}: "
                    f"не найден файл конфигурации"
                )
                continue

            try:
                self._voices[voice_name] = PiperVoice.load(
                    str(model_path),
                    config_path=str(config_path),
                )
                print(f"Загружена модель: {voice_name}")
            except Exception as exc:
                print(f"Ошибка загрузки {model_path.name}: {exc}")

        if not self._voices:
            print("Не удалось загрузить ни одной модели Piper TTS.")

    def voices(self) -> list[str]:
        return list(self._voices)

    def synthesize(self, voice: str, text: str) -> bytes:
        piper_voice = self._voices.get(voice)

        if piper_voice is None:
            raise KeyError(voice)

        audio = io.BytesIO()

        with wave.open(audio, "wb") as wav_file:
            piper_voice.synthesize_wav(text, wav_file)

        return audio.getvalue()
