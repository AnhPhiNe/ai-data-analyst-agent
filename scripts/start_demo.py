from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence


def main() -> int:
    streamlit_port = os.getenv("PORT", "8501")
    env = {
        **os.environ,
        "BACKEND_URL": os.getenv("BACKEND_URL", "http://127.0.0.1:8000"),
    }
    processes = [
        _start(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
            env,
        ),
        _start(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "frontend/streamlit_app.py",
                "--server.address",
                "0.0.0.0",
                "--server.port",
                streamlit_port,
                "--server.headless",
                "true",
            ],
            env,
        ),
    ]

    def stop_processes(*_: object) -> None:
        for process in processes:
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGTERM, stop_processes)
    signal.signal(signal.SIGINT, stop_processes)

    try:
        while True:
            for process in processes:
                exit_code = process.poll()
                if exit_code is not None:
                    stop_processes()
                    return int(exit_code)
            time.sleep(1)
    finally:
        stop_processes()


def _start(command: Sequence[str], env: dict[str, str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
