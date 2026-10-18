import io
from pathlib import Path
from typing import Any, BinaryIO

from verity_ai.config import (
    STT_BEAM_SIZE,
    STT_COMPUTE_TYPE,
    STT_DEVICE,
    STT_LANGUAGE,
    STT_MODEL,
    STT_MODEL_DIR,
)
from verity_ai.core.stt import (
    STTProvider,
    TranscriptionResult,
    TranscriptionSegment,
)

SUPPORTED_MODELS = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "distil-small.en",
    "distil-medium.en",
    "distil-large-v2",
    "distil-large-v3",
)

OPENAI_ALIASES = {"whisper-1"}


class WhisperProvider(STTProvider):
    name = "whisper"
    title = "Whisper"
    description = "Локальное распознавание речи faster-whisper"

    def __init__(
        self,
        default_model: str = STT_MODEL,
        model_dir: Path = STT_MODEL_DIR,
        device: str = STT_DEVICE,
        compute_type: str = STT_COMPUTE_TYPE,
        beam_size: int = STT_BEAM_SIZE,
        language: str | None = STT_LANGUAGE or None,
    ):
        self.default_model = default_model
        self.model_dir = model_dir
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.language = language
        self._models: dict[str, Any] = {}

    def models(self) -> list[str]:
        return sorted({*SUPPORTED_MODELS, "whisper-1", self.default_model})

    def supports(self, model: str) -> bool:
        name = (model or "").strip()

        if not name:
            return False

        return (
            name in OPENAI_ALIASES
            or name.lower() in SUPPORTED_MODELS
            or name == self.default_model
        )

    def resolve_model(self, model: str) -> str:
        name = (model or "").strip()

        if not name or name in OPENAI_ALIASES:
            return self.default_model

        return name

    def _load_model(self, model: str) -> Any:
        cached = self._models.get(model)

        if cached is not None:
            return cached

        from faster_whisper import WhisperModel

        print(f"Загрузка модели faster-whisper: {model}")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        instance = WhisperModel(
            model,
            device=self.device,
            compute_type=self.compute_type,
            download_root=str(self.model_dir),
        )
        self._models[model] = instance

        return instance

    def transcribe(
        self,
        model: str,
        audio: BinaryIO | str | Path,
        language: str | None = None,
    ) -> TranscriptionResult:
        model_name = self.resolve_model(model)
        instance = self._load_model(model_name)

        source: BinaryIO | str
        if isinstance(audio, bytes):
            source = io.BytesIO(audio)
        elif isinstance(audio, Path):
            source = str(audio)
        else:
            source = audio

        segments, info = instance.transcribe(
            source,
            language=language or self.language,
            beam_size=self.beam_size,
        )

        transcribed: list[TranscriptionSegment] = []
        texts: list[str] = []

        for segment in segments:
            text = segment.text.strip()
            transcribed.append(
                TranscriptionSegment(
                    id=segment.id,
                    start=segment.start,
                    end=segment.end,
                    text=text,
                )
            )

            if text:
                texts.append(text)

        return TranscriptionResult(
            text=" ".join(texts).strip(),
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            duration=getattr(info, "duration", None),
            segments=tuple(transcribed),
        )
