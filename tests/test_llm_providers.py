import importlib

import pytest
import requests
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

deepseek_module = importlib.import_module(
    "verity_ai.modules.llm_modules.deepseek.provider"
)
ollama_module = importlib.import_module(
    "verity_ai.modules.llm_modules.ollama.provider"
)


class FakeResponse:
    def __init__(
        self,
        payload=None,
        lines=(),
        status_code=200,
        text="",
    ):
        self.payload = payload
        self.lines = list(lines)
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self.payload

    def iter_lines(self):
        return iter(self.lines)


def test_deepseek_metadata():
    provider = deepseek_module.DeepSeekProvider()

    assert provider.supports("deepseek-flash")
    assert provider.supports("deepseek-chat")
    assert not provider.supports("gpt-4")
    assert provider.resolve_model("deepseek-coder") == "deepseek-flash"
    assert provider.resolve_model("unknown") is None
    assert provider.tags() == []
    assert provider.models()[0]["id"] == "deepseek-flash"
    assert provider.models()[0]["owned_by"] == "deepseek"


def test_deepseek_chat_requires_api_key(monkeypatch):
    monkeypatch.setattr(deepseek_module, "DEEPSEEK_API_KEY", "")
    provider = deepseek_module.DeepSeekProvider()

    with pytest.raises(HTTPException) as error:
        provider.chat({"model": "deepseek-flash", "messages": []})

    assert error.value.status_code == 500
    assert "DEEPSEEK_API_KEY" in error.value.detail


def test_deepseek_chat_rejects_unknown_model():
    provider = deepseek_module.DeepSeekProvider()

    with pytest.raises(HTTPException) as error:
        provider.chat({"model": "unknown", "messages": []})

    assert error.value.status_code == 400


def test_deepseek_chat_sends_payload_and_returns_json(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse(payload={"id": "response"})

    monkeypatch.setattr(deepseek_module.requests, "post", fake_post)
    monkeypatch.setattr(deepseek_module, "DEEPSEEK_API_KEY", "secret")
    provider = deepseek_module.DeepSeekProvider()

    result = provider.chat(
        {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "max_tokens": 10,
            "unknown": "ignored",
        }
    )

    assert result == {"id": "response"}
    assert captured["url"] == deepseek_module.DEEPSEEK_URL
    assert captured["json"] == {
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 10,
    }
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["headers"]["Content-Type"] == "application/json"
    assert captured["stream"] is False


def test_deepseek_chat_returns_streaming_response(monkeypatch):
    def fake_post(url, **kwargs):
        return FakeResponse(lines=[b"data: one", b"data: two"])

    monkeypatch.setattr(deepseek_module.requests, "post", fake_post)
    provider = deepseek_module.DeepSeekProvider()

    result = provider.chat({"model": "deepseek-flash", "stream": True})

    assert isinstance(result, StreamingResponse)
    assert result.media_type == "text/event-stream"


def test_deepseek_chat_maps_http_error(monkeypatch):
    def fake_post(url, **kwargs):
        return FakeResponse(status_code=429, text="rate limited")

    monkeypatch.setattr(deepseek_module.requests, "post", fake_post)
    provider = deepseek_module.DeepSeekProvider()

    with pytest.raises(HTTPException) as error:
        provider.chat({"model": "deepseek-flash", "messages": []})

    assert error.value.status_code == 429
    assert error.value.detail == "rate limited"


def test_deepseek_chat_maps_connection_error(monkeypatch):
    def fake_post(url, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(deepseek_module.requests, "post", fake_post)
    provider = deepseek_module.DeepSeekProvider()

    with pytest.raises(HTTPException) as error:
        provider.chat({"model": "deepseek-flash", "messages": []})

    assert error.value.status_code == 502
    assert error.value.detail == "offline"


def test_ollama_metadata_and_prefix_resolution():
    provider = ollama_module.OllamaProvider()

    assert provider.supports("qwen1")
    assert provider.supports("ollama-qwen1")
    assert not provider.supports("unknown")
    assert provider.resolve_model("qwen1") == "qwen2.5:3b"
    assert provider.resolve_model("ollama-llama") == "llama3.2"
    assert provider.resolve_model("unknown") is None
    assert {"name": "qwen1", "model": "qwen2.5:3b"} in provider.tags()
    assert len(provider.models()) == 2


def test_ollama_chat_transforms_response(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse(
            payload={
                "message": {"role": "assistant", "content": "hello"},
            }
        )

    monkeypatch.setattr(ollama_module.requests, "post", fake_post)
    provider = ollama_module.OllamaProvider()

    result = provider.chat(
        {
            "model": "qwen1",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.1,
        }
    )

    assert result["id"] == "ollama-chat"
    assert result["object"] == "chat.completion"
    assert result["model"] == "qwen2.5:3b"
    assert result["choices"][0]["message"]["content"] == "hello"
    assert captured["url"] == ollama_module.OLLAMA_URL
    assert captured["json"]["options"] == {"temperature": 0.1}


def test_ollama_chat_returns_streaming_response(monkeypatch):
    def fake_post(url, **kwargs):
        return FakeResponse(lines=[b'{"message": "one"}'])

    monkeypatch.setattr(ollama_module.requests, "post", fake_post)
    provider = ollama_module.OllamaProvider()

    result = provider.chat({"model": "llama", "stream": True})

    assert isinstance(result, StreamingResponse)
    assert result.media_type == "application/x-ndjson"


def test_ollama_chat_maps_error(monkeypatch):
    def fake_post(url, **kwargs):
        return FakeResponse(status_code=500, text="crashed")

    monkeypatch.setattr(ollama_module.requests, "post", fake_post)
    provider = ollama_module.OllamaProvider()

    with pytest.raises(HTTPException) as error:
        provider.chat({"model": "llama", "messages": []})

    assert error.value.status_code == 500
    assert "Ollama недоступна: crashed" in error.value.detail


def test_ollama_chat_rejects_unknown_model():
    provider = ollama_module.OllamaProvider()

    with pytest.raises(HTTPException) as error:
        provider.chat({"model": "unknown", "messages": []})

    assert error.value.status_code == 400
