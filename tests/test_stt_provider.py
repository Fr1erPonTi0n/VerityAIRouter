import pytest

import verity_ai.modules.stt_modules.whisper.provider as whisper_module


class FakeSegment:
    def __init__(self, id, start, end, text):
        self.id = id
        self.start = start
        self.end = end
        self.text = text


class FakeInfo:
    language = "ru"
    language_probability = 0.98
    duration = 1.5


class FakeWhisperModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, language=None, beam_size=5):
        self.calls.append((audio, language, beam_size))
        return iter(
            [
                FakeSegment(0, 0.0, 1.0, " Привет "),
                FakeSegment(1, 1.0, 1.5, " мир "),
            ]
        ), FakeInfo()


def test_whisper_models_and_aliases(tmp_path):
    provider = whisper_module.WhisperProvider(
        default_model="small",
        model_dir=tmp_path,
    )

    assert provider.supports("whisper-1")
    assert provider.supports("small")
    assert provider.supports("SMALL")
    assert not provider.supports("missing")
    assert provider.resolve_model("whisper-1") == "small"
    assert provider.resolve_model("") == "small"
    assert provider.resolve_model("base") == "base"
    assert {"whisper-1", "small", "large-v3"} <= set(provider.models())


def test_whisper_transcribe_collects_segments(tmp_path, monkeypatch):
    provider = whisper_module.WhisperProvider(
        default_model="small",
        model_dir=tmp_path,
        beam_size=3,
    )
    instance = FakeWhisperModel()
    loaded = []

    monkeypatch.setattr(
        provider,
        "_load_model",
        lambda name: loaded.append(name) or instance,
    )

    result = provider.transcribe(
        "whisper-1",
        tmp_path / "audio.wav",
        language="ru",
    )

    assert loaded == ["small"]
    assert result.text == "Привет мир"
    assert result.language == "ru"
    assert result.language_probability == pytest.approx(0.98)
    assert result.duration == pytest.approx(1.5)
    assert [(segment.id, segment.text) for segment in result.segments] == [
        (0, "Привет"),
        (1, "мир"),
    ]
    assert instance.calls == [
        (str(tmp_path / "audio.wav"), "ru", 3),
    ]


def test_whisper_transcribe_uses_configured_language(tmp_path, monkeypatch):
    provider = whisper_module.WhisperProvider(
        default_model="small",
        model_dir=tmp_path,
        language="en",
    )
    instance = FakeWhisperModel()

    monkeypatch.setattr(provider, "_load_model", lambda name: instance)

    provider.transcribe("small", tmp_path / "audio.wav")

    assert instance.calls[0][1] == "en"


def test_whisper_caches_loaded_models(tmp_path, monkeypatch):
    provider = whisper_module.WhisperProvider(
        default_model="small",
        model_dir=tmp_path,
    )
    created = []

    class FakeModel:
        def __init__(self, name, **kwargs):
            created.append((name, kwargs))

    monkeypatch.setattr(
        "faster_whisper.WhisperModel",
        FakeModel,
    )

    first = provider._load_model("small")
    second = provider._load_model("small")

    assert first is second
    assert created == [
        (
            "small",
            {
                "device": "auto",
                "compute_type": "int8",
                "download_root": str(tmp_path),
            },
        )
    ]
