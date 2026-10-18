import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from verity_ai.config import STT_MODEL
from verity_ai.core.stt.providers import (
    available_stt_models,
    find_stt_provider,
)

router = APIRouter()

RESPONSE_FORMATS = ("json", "text", "verbose_json")


@router.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form(STT_MODEL),
    language: str | None = Form(None),
    response_format: str = Form("json"),
):
    model_name = (model or STT_MODEL).strip()
    provider = find_stt_provider(model_name)

    if provider is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Неизвестная модель '{model_name}'. Доступные: "
                f"{', '.join(available_stt_models())}."
            ),
        )

    fmt = (response_format or "json").strip().lower()

    if fmt not in RESPONSE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Неподдерживаемый response_format '{response_format}'. "
                f"Доступны: {', '.join(RESPONSE_FORMATS)}."
            ),
        )

    suffix = Path(file.filename or "").suffix or ".wav"
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            prefix="verity-stt-",
            suffix=suffix,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

            while chunk := await file.read(1024 * 1024):
                temp_file.write(chunk)

        if temp_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Аудиофайл пуст.")

        try:
            result = provider.transcribe(model_name, temp_path, language or None)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"STT Error: {exc}")
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    if fmt == "text":
        return PlainTextResponse(result.text)

    if fmt == "verbose_json":
        return {
            "task": "transcribe",
            "language": result.language,
            "duration": result.duration,
            "text": result.text,
            "segments": [
                {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
                for segment in result.segments
            ],
        }

    return {"text": result.text}


@router.get("/v1/audio/transcriptions/models")
async def get_stt_models():
    return {"models": available_stt_models()}
