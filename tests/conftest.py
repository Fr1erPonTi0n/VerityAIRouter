import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TTS_MODEL_DIR = tempfile.mkdtemp(prefix="verity-tts-")
atexit.register(shutil.rmtree, TTS_MODEL_DIR, ignore_errors=True)

os.environ["TTS_MODEL_DIR"] = TTS_MODEL_DIR
os.environ["DEEPSEEK_API_KEY"] = "test-deepseek-key"
os.environ["TTS_API_URL"] = ""
os.environ["TTS_API_KEY"] = ""
os.environ["TTS_API_VOICES"] = ""
os.environ["TTS_API_VOICE"] = "default"


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from verity_ai.app import app

    with TestClient(app) as test_client:
        yield test_client
