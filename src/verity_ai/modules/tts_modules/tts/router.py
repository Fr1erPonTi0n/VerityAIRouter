import io

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from verity_ai.config import DEFAULT_TTS_VOICE
from verity_ai.core.tts.voices import available_voices, find_voice

router = APIRouter()


class TTSRequest(BaseModel):
    input: str
    voice: str = DEFAULT_TTS_VOICE
    response_format: str = "wav"


@router.post("/v1/audio/speech")
async def generate_audio(req: TTSRequest):
    text = req.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Текст не должен быть пустым.")

    found = find_voice(req.voice)
    if found is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Голос '{req.voice}' не найден. "
                f"Доступны: {', '.join(available_voices())}"
            ),
        )

    provider, voice = found

    try:
        audio = provider.synthesize(voice, text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS Error: {exc}")

    return StreamingResponse(io.BytesIO(audio), media_type="audio/wav")


@router.get("/v1/audio/voices")
async def get_voices():
    return {"voices": available_voices()}
