"""SecureDoc Core 통합 실행 스크립트.

`python run.py` 한 번으로 FastAPI(8000) + Streamlit UI(8501)를 함께 띄웁니다.
Ctrl+C 한 번이면 두 프로세스가 함께 정리됩니다.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    print("=" * 64)
    print(" SecureDoc Core 통합 실행")
    print("  - FastAPI : http://localhost:8000   (API + /docs)")
    print("  - UI      : http://localhost:8501   (Streamlit)")
    print(" 종료하려면 Ctrl+C")
    print("=" * 64, flush=True)

    api = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "app.main:app",
            "--host", "0.0.0.0", "--port", "8000",
        ],
        cwd=ROOT,
    )
    ui = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "ui/app.py",
            "--server.port", "8501",
        ],
        cwd=ROOT,
    )

    procs = [("FastAPI", api), ("Streamlit", ui)]

    try:
        while True:
            for name, p in procs:
                code = p.poll()
                if code is not None:
                    print(
                        f"\n[{name}] 프로세스가 종료됨 (exit={code}). "
                        "나머지도 정리합니다.",
                        flush=True,
                    )
                    raise KeyboardInterrupt
            try:
                api.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
    except KeyboardInterrupt:
        print("\n종료 신호 감지 — 프로세스 정리 중...", flush=True)
    finally:
        for _, p in procs:
            if p.poll() is None:
                p.terminate()
        for _, p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

    return 0


if __name__ == "__main__":
    sys.exit(main())
