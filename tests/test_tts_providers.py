import io
import wave

import pytest
import requests

import verity_ai.modules.tts_modules.piper.provider as piper_module
import verity_ai.modules.tts_modules.remote.provider as remote_module
from verity_ai.core.tts.api import APIVoiceProvider, TTSAPIError


class FakeHTTPResponse:
    def __init__(self, payload=None, content=b"", status_code=200, text=""):
        self.payload = payload
        self.content = content
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self.payload


class RecordingSession:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.trust_env = True

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.response


class FailingSession:
    def __init__(self, error):
        self.error = error
        self.trust_env = True

    def request(self, method, url, **kwargs):
        raise self.error


class DummyAPIVoiceProvider(APIVoiceProvider):
    def voices(self):
        return []

    def synthesize(self, voice, text):
        return b""


class FakePiperVoice:
    def synthesize_wav(self, text, wav_file):
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"\x00\x00" * 8)


def test_piper_loads_models_and_synthesizes(tmp_path, monkeypatch, capsys):
    (tmp_path / "alice.onnx").write_bytes(b"model")
    (tmp_path / "alice.json").write_text("{}", encoding="utf-8")
    (tmp_path / "broken.onnx").write_bytes(b"model")

    monkeypatch.setattr(
        piper_module.PiperVoice,
        "load",
        lambda *args, **kwargs: FakePiperVoice(),
    )

    provider = piper_module.PiperProvider(tmp_path)

    assert provider.voices() == ["alice"]
    assert "Пропуск broken.onnx" in capsys.readouterr().out

    audio = provider.synthesize("alice", "hello")

    with wave.open(io.BytesIO(audio), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 22050
        assert wav_file.getnframes() == 8


def test_piper_synthesize_rejects_unknown_voice(tmp_path):
    provider = piper_module.PiperProvider(tmp_path)

    with pytest.raises(KeyError):
        provider.synthesize("missing", "hello")


def test_api_provider_headers_and_url():
    provider = DummyAPIVoiceProvider()
    provider.url = "http://tts.local/v1/"
    provider.api_key = "secret"

    assert provider._headers() == {
        "Accept": "audio/wav",
        "Content-Type": "application/json",
        "Authorization": "Bearer secret",
    }
    assert provider._url("/audio") == "http://tts.local/v1/audio"

    provider.api_key = ""

    assert "Authorization" not in provider._headers()


def test_api_provider_posts_json():
    response = FakeHTTPResponse(payload={"ok": True})
    session = RecordingSession(response)
    provider = DummyAPIVoiceProvider()
    provider._session = session
    provider.timeout = 5

    result = provider._post_json(
        "http://tts.local/audio",
        {"input": "hello"},
    )

    assert result is response
    method, url, kwargs = session.calls[0]
    assert method == "POST"
    assert url == "http://tts.local/audio"
    assert kwargs["json"] == {"input": "hello"}
    assert kwargs["timeout"] == 5


def test_api_provider_gets_json():
    response = FakeHTTPResponse(payload={"voices": ["alice"]})
    session = RecordingSession(response)
    provider = DummyAPIVoiceProvider()
    provider._session = session

    result = provider._get_json(
        "http://tts.local/voices",
        params={"limit": 1},
    )

    assert result == {"voices": ["alice"]}
    method, url, kwargs = session.calls[0]
    assert method == "GET"
    assert url == "http://tts.local/voices"
    assert kwargs["params"] == {"limit": 1}


def test_api_provider_wraps_connection_error():
    provider = DummyAPIVoiceProvider()
    provider._session = FailingSession(requests.ConnectionError("offline"))

    with pytest.raises(TTSAPIError, match="offline"):
        provider._post_json("http://tts.local/audio", {})


def test_api_provider_wraps_http_error():
    response = FakeHTTPResponse(status_code=503, text="unavailable")
    provider = DummyAPIVoiceProvider()
    provider._session = RecordingSession(response)

    with pytest.raises(TTSAPIError, match="unavailable"):
        provider._post_json("http://tts.local/audio", {})


def test_remote_provider_voices_from_config(monkeypatch):
    monkeypatch.setattr(remote_module, "TTS_API_VOICES", "alice, bob,")
    monkeypatch.setattr(remote_module, "TTS_API_VOICE", "default")
    provider = remote_module.RemoteTTSProvider()

    assert provider.voices() == ["alice", "bob"]


def test_remote_provider_falls_back_to_default_voice(monkeypatch):
    monkeypatch.setattr(remote_module, "TTS_API_VOICES", "")
    monkeypatch.setattr(remote_module, "TTS_API_VOICE", "default")
    provider = remote_module.RemoteTTSProvider()

    assert provider.voices() == ["default"]


def test_remote_provider_synthesizes_audio(monkeypatch):
    monkeypatch.setattr(remote_module, "TTS_API_URL", "http://tts.local")
    monkeypatch.setattr(remote_module, "TTS_API_VOICES", "")
    monkeypatch.setattr(remote_module, "TTS_API_VOICE", "default")
    provider = remote_module.RemoteTTSProvider()
    calls = []

    def fake_post(url, payload):
        calls.append((url, payload))
        return FakeHTTPResponse(content=b"audio")

    monkeypatch.setattr(provider, "_post_json", fake_post)

    assert provider.synthesize("alice", "hello") == b"audio"
    assert calls == [
        (
            "http://tts.local",
            {
                "input": "hello",
                "voice": "alice",
                "response_format": "wav",
            },
        )
    ]


def test_remote_provider_rejects_empty_audio(monkeypatch):
    monkeypatch.setattr(remote_module, "TTS_API_URL", "http://tts.local")
    monkeypatch.setattr(remote_module, "TTS_API_VOICES", "")
    monkeypatch.setattr(remote_module, "TTS_API_VOICE", "default")
    provider = remote_module.RemoteTTSProvider()
    monkeypatch.setattr(
        provider,
        "_post_json",
        lambda url, payload: FakeHTTPResponse(content=b""),
    )

    with pytest.raises(TTSAPIError, match="пустой ответ"):
        provider.synthesize("alice", "hello")
