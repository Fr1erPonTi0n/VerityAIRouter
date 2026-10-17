import importlib.util
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_create_env():
    spec = importlib.util.spec_from_file_location(
        "create_env_script",
        ROOT / "scripts" / "create_env.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


create_env = load_create_env()


def test_read_env_ignores_comments_and_blank_lines(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "# comment\n\nA=1\nB = \"two\"\nC='three'\ninvalid line\n",
        encoding="utf-8",
    )

    assert create_env.read_env(path) == {
        "A": "1",
        "B": "two",
        "C": "three",
    }


def test_read_env_returns_empty_dict_for_missing_file(tmp_path):
    assert create_env.read_env(tmp_path / "missing.env") == {}


def test_build_content_contains_sections_and_extras():
    content = create_env.build_content(
        create_env.FIELDS,
        {"EXTRA": "value"},
    )

    assert content.startswith("# Verity AI Router")
    assert "HOST=127.0.0.1" in content
    assert "DEFAULT_MODEL=deepseek-flash" in content
    assert "# Дополнительные переменные" in content
    assert "EXTRA=value" in content
    assert content.endswith("\n")


def test_ask_uses_environment_for_non_interactive_input(monkeypatch):
    monkeypatch.setattr(create_env.sys, "stdin", io.StringIO())
    monkeypatch.setenv("VERITY_ASK_VALUE", "from-env")

    assert create_env.ask(
        "VERITY_ASK_VALUE",
        "default",
        secret=False,
    ) == "from-env"


def test_ask_falls_back_to_default_for_non_interactive_input(monkeypatch):
    monkeypatch.setattr(create_env.sys, "stdin", io.StringIO())
    monkeypatch.delenv("VERITY_ASK_VALUE", raising=False)

    assert create_env.ask(
        "VERITY_ASK_VALUE",
        "default",
        secret=False,
    ) == "default"


def test_main_creates_env_file(tmp_path, monkeypatch):
    target = tmp_path / ".env"
    monkeypatch.setattr(create_env, "ENV_PATH", target)
    monkeypatch.setattr(
        create_env,
        "ask",
        lambda name, default, secret: default,
    )

    assert create_env.main() == 0
    assert target.exists()
    assert "HOST=127.0.0.1" in target.read_text(encoding="utf-8")


def test_main_does_not_overwrite_without_force(tmp_path, monkeypatch):
    target = tmp_path / ".env"
    target.write_text("OLD", encoding="utf-8")
    monkeypatch.setattr(create_env, "ENV_PATH", target)

    assert create_env.main() == 1
    assert target.read_text(encoding="utf-8") == "OLD"


def test_main_overwrites_with_force_and_keeps_extras(tmp_path, monkeypatch):
    target = tmp_path / ".env"
    target.write_text("CUSTOM_VALUE=42\n", encoding="utf-8")
    monkeypatch.setattr(create_env, "ENV_PATH", target)
    monkeypatch.setattr(create_env.sys, "argv", ["create_env.py", "--force"])
    monkeypatch.setattr(
        create_env,
        "ask",
        lambda name, default, secret: default,
    )

    assert create_env.main() == 0
    content = target.read_text(encoding="utf-8")
    assert "CUSTOM_VALUE=42" in content
    assert "HOST=127.0.0.1" in content
