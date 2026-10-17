import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from verity_ai.config import HOST, PORT


if __name__ == "__main__":
    print(f"Verity AI Router: http://{HOST}:{PORT}")
    print("Для остановки нажмите Ctrl+C.")

    uvicorn.run(
        "verity_ai.app:app",
        host=HOST,
        port=PORT,
        app_dir=str(ROOT / "src"),
    )
