from verity_ai import registry


def test_iter_module_names_finds_all_packages():
    names = set(registry.iter_module_names())

    assert "verity_ai.modules.llm_modules.deepseek" in names
    assert "verity_ai.modules.llm_modules.ollama" in names
    assert "verity_ai.modules.llm_modules.llm" in names
    assert "verity_ai.modules.tts_modules.piper" in names
    assert "verity_ai.modules.tts_modules.tts" in names


def test_discover_modules_builds_entries():
    modules = {module.name: module for module in registry.discover_modules()}

    assert {"deepseek", "ollama", "llm", "piper", "tts"} <= set(modules)
    assert modules["deepseek"].provider is not None
    assert modules["deepseek"].kind == "llm"
    assert modules["llm"].router is not None
    assert modules["piper"].voice_provider is not None
    assert modules["tts"].kind == "tts"


def test_modules_endpoint_returns_discovered_modules(client):
    response = client.get("/api/ai/modules")

    assert response.status_code == 200
    modules = {module["name"]: module for module in response.json()["modules"]}
    assert {"deepseek", "ollama", "llm", "piper", "tts"} <= set(modules)
    assert modules["deepseek"]["provider"] is True
    assert modules["llm"]["router"] is True
    assert modules["piper"]["voice_provider"] is True
