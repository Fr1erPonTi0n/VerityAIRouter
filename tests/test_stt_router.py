import importlib

from verity_ai.core.stt import TranscriptionResult, TranscriptionSegment

stt_router = importlib.import_module("verity_ai.modules.stt_modules.stt.router")


class FakeSTTProvider:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def transcribe(self, model, audio, language=None):
        self.calls.append((model, audio.read_bytes(), language))

        if self.error is not None:
            raise self.error

        return TranscriptionResult(
            text="hello world",
            language="en",
            language_probability=0.9,
            duration=2.0,
            segments=(
                TranscriptionSegment(id=0, start=0.0, end=1.0, text="hello"),
                TranscriptionSegment(id=1, start=1.0, end=2.0, text="world"),
            ),
        )


def test_transcription_returns_json(client, monkeypatch):
    provider = FakeSTTProvider()
    monkeypatch.setattr(
        stt_router,
        "find_stt_provider",
        lambda model: provider,
    )

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", b"audio-bytes", "audio/wav")},
        data={"model": "whisper-1", "language": "en"},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "hello world"}
    assert provider.calls == [("whisper-1", b"audio-bytes", "en")]


def test_transcription_returns_text(client, monkeypatch):
    monkeypatch.setattr(
        stt_router,
        "find_stt_provider",
        lambda model: FakeSTTProvider(),
    )

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", b"audio-bytes", "audio/wav")},
        data={"model": "whisper-1", "response_format": "text"},
    )

    assert response.status_code == 200
    assert response.text == "hello world"
    assert response.headers["content-type"].startswith("text/plain")


def test_transcription_returns_verbose_json(client, monkeypatch):
    monkeypatch.setattr(
        stt_router,
        "find_stt_provider",
        lambda model: FakeSTTProvider(),
    )

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", b"audio-bytes", "audio/wav")},
        data={"model": "whisper-1", "response_format": "verbose_json"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "task": "transcribe",
        "language": "en",
        "duration": 2.0,
        "text": "hello world",
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "hello"},
            {"id": 1, "start": 1.0, "end": 2.0, "text": "world"},
        ],
    }


def test_transcription_rejects_unknown_model(client, monkeypatch):
    monkeypatch.setattr(stt_router, "find_stt_provider", lambda model: None)
    monkeypatch.setattr(
        stt_router,
        "available_stt_models",
        lambda: ["small", "whisper-1"],
    )

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", b"audio-bytes", "audio/wav")},
        data={"model": "missing"},
    )

    assert response.status_code == 400
    assert "missing" in response.json()["detail"]
    assert "small, whisper-1" in response.json()["detail"]


def test_transcription_rejects_unknown_response_format(client, monkeypatch):
    monkeypatch.setattr(
        stt_router,
        "find_stt_provider",
        lambda model: FakeSTTProvider(),
    )

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", b"audio-bytes", "audio/wav")},
        data={"model": "whisper-1", "response_format": "srt"},
    )

    assert response.status_code == 400
    assert "srt" in response.json()["detail"]


def test_transcription_rejects_empty_file(client, monkeypatch):
    monkeypatch.setattr(
        stt_router,
        "find_stt_provider",
        lambda model: FakeSTTProvider(),
    )

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", b"", "audio/wav")},
        data={"model": "whisper-1"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Аудиофайл пуст."


def test_transcription_maps_provider_error(client, monkeypatch):
    monkeypatch.setattr(
        stt_router,
        "find_stt_provider",
        lambda model: FakeSTTProvider(error=RuntimeError("boom")),
    )

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", b"audio-bytes", "audio/wav")},
        data={"model": "whisper-1"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "STT Error: boom"


def test_stt_models_endpoint(client, monkeypatch):
    monkeypatch.setattr(
        stt_router,
        "available_stt_models",
        lambda: ["small", "whisper-1"],
    )

    response = client.get("/v1/audio/transcriptions/models")

    assert response.status_code == 200
    assert response.json() == {"models": ["small", "whisper-1"]}
