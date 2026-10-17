import importlib

import pytest

from verity_ai.config import DEFAULT_MODEL

chat_router = importlib.import_module(
    "verity_ai.modules.llm_modules.llm.router"
)


class FakeProvider:
    def __init__(self):
        self.calls = []

    def chat(self, data):
        self.calls.append(data)
        return {"id": "fake", "model": data.get("model", "default")}


@pytest.mark.parametrize(
    "path",
    ["/api/chat", "/v1/chat/completions", "/chat/completions"],
)
def test_chat_routes_dispatch_to_provider(client, monkeypatch, path):
    provider = FakeProvider()
    seen = []

    def fake_find(model):
        seen.append(model)
        return provider

    monkeypatch.setattr(chat_router, "find_provider", fake_find)

    response = client.post(path, json={"model": "FAKE", "messages": []})

    assert response.status_code == 200
    assert response.json() == {"id": "fake", "model": "FAKE"}
    assert seen == ["fake"]


def test_chat_uses_default_model(client, monkeypatch):
    provider = FakeProvider()
    seen = []

    def fake_find(model):
        seen.append(model)
        return provider

    monkeypatch.setattr(chat_router, "find_provider", fake_find)

    response = client.post("/api/chat", json={"messages": []})

    assert response.status_code == 200
    assert seen == [DEFAULT_MODEL]


def test_chat_rejects_invalid_json(client):
    response = client.post(
        "/api/chat",
        content="{not-json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid JSON payload"


def test_chat_rejects_unknown_model(client, monkeypatch):
    monkeypatch.setattr(chat_router, "find_provider", lambda model: None)
    monkeypatch.setattr(
        chat_router,
        "available_models",
        lambda: ["deepseek-flash", "llama"],
    )

    response = client.post("/api/chat", json={"model": "missing"})

    assert response.status_code == 400
    assert "deepseek-flash, llama" in response.json()["detail"]


def test_models_endpoint_lists_known_models(client):
    response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    ids = {model["id"] for model in payload["data"]}
    assert {"deepseek-flash", "qwen", "llama"} <= ids


def test_tags_endpoint_lists_ollama_tags(client):
    response = client.get("/api/tags")

    assert response.status_code == 200
    tags = response.json()["models"]
    assert {"name": "qwen1", "model": "qwen2.5:3b"} in tags
