import importlib

import pytest

from verity_ai.core.tts.voice import VoiceProvider

tts_router = importlib.import_module("verity_ai.modules.tts_modules.tts.router")


class FakeVoiceProvider(VoiceProvider):
    name = "fake"
    title = "Fake"

    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def voices(self):
        return ["alice"]

    def synthesize(self, voice, text):
        self.calls.append((voice, text))

        if self.error is not None:
            raise self.error

        return f"{voice}:{text}".encode()


def test_speech_returns_wav(client, monkeypatch):
    provider = FakeVoiceProvider()
    monkeypatch.setattr(
        tts_router,
        "find_voice",
        lambda voice: (provider, voice),
    )
    monkeypatch.setattr(tts_router, "available_voices", lambda: ["alice"])

    response = client.post(
        "/v1/audio/speech",
        json={"input": "  hello  ", "voice": "alice"},
    )

    assert response.status_code == 200
    assert response.content == b"alice:hello"
    assert response.headers["content-type"].startswith("audio/wav")
    assert provider.calls == [("alice", "hello")]


def test_speech_rejects_empty_text(client):
    response = client.post(
        "/v1/audio/speech",
        json={"input": "   ", "voice": "alice"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Текст не должен быть пустым."


def test_speech_rejects_unknown_voice(client, monkeypatch):
    monkeypatch.setattr(tts_router, "find_voice", lambda voice: None)
    monkeypatch.setattr(tts_router, "available_voices", lambda: ["alice", "bob"])

    response = client.post(
        "/v1/audio/speech",
        json={"input": "hello", "voice": "missing"},
    )

    assert response.status_code == 404
    assert "missing" in response.json()["detail"]
    assert "alice, bob" in response.json()["detail"]


def test_speech_maps_provider_error(client, monkeypatch):
    provider = FakeVoiceProvider(error=RuntimeError("boom"))
    monkeypatch.setattr(
        tts_router,
        "find_voice",
        lambda voice: (provider, voice),
    )

    response = client.post(
        "/v1/audio/speech",
        json={"input": "hello", "voice": "alice"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "TTS Error: boom"


def test_voices_endpoint_returns_available_voices(client, monkeypatch):
    monkeypatch.setattr(tts_router, "available_voices", lambda: ["bob", "alice"])

    response = client.get("/v1/audio/voices")

    assert response.status_code == 200
    assert response.json() == {"voices": ["bob", "alice"]}
