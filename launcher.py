"""Windows-safe launcher: batch does not parse Python commands or branch blocks."""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEPENDENCY_CHECK = "import streamlit, openpyxl, pandas, pydantic, httpx, jsonschema, pypdf, docx"


def find_python() -> Path:
    candidates = [ROOT/".venv"/"Scripts"/"python.exe",
                  ROOT.parent/".venv-judge"/"Scripts"/"python.exe",
                  ROOT/".venv"/"bin"/"python"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    local = ROOT/".venv"
    print("正在准备独立运行环境…", flush=True)
    subprocess.run([sys.executable, "-m", "venv", str(local)], cwd=ROOT, check=True)
    return local/("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def open_when_ready(url: str, process: subprocess.Popen):
    for _ in range(100):
        if process.poll() is not None:
            return
        try:
            with urllib.request.urlopen(url+"/_stcore/health", timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except (OSError, ValueError):
            pass
        time.sleep(0.3)


def main() -> int:
    parser = argparse.ArgumentParser(description="启动本地裁判工作台")
    parser.add_argument("--check", action="store_true", help="仅验证运行环境，不启动网页或浏览器")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        print("需要 Python 3.11 或以上版本。")
        return 1
    if not 1 <= args.port <= 65535:
        print("端口需在1至65535之间。")
        return 1
    try:
        interpreter = find_python()
        check = subprocess.run([str(interpreter), "-c", DEPENDENCY_CHECK], cwd=ROOT,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if check.returncode:
            print("正在安装依赖，首次启动需要联网，请稍候…", flush=True)
            subprocess.run([str(interpreter), "-m", "pip", "install", "-r", str(ROOT/"requirements.txt")],
                           cwd=ROOT, check=True)
            subprocess.run([str(interpreter), "-c", DEPENDENCY_CHECK], cwd=ROOT, check=True)
        if args.check:
            print("JUDGE_LAUNCHER_OK", flush=True)
            return 0
        url = f"http://127.0.0.1:{args.port}"
        print(f"正在启动裁判工作台：{url}\n使用期间请保留此窗口；关闭窗口即可停止。", flush=True)
        process = subprocess.Popen([str(interpreter), "-m", "streamlit", "run", str(ROOT/"app.py"),
                                    "--server.address", "127.0.0.1", "--server.port", str(args.port),
                                    "--server.headless", "true"], cwd=ROOT)
        if not args.no_browser:
            threading.Thread(target=open_when_ready, args=(url, process), daemon=True).start()
        try:
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            return 0
    except (OSError, subprocess.CalledProcessError):
        print("启动失败。请检查Python安装、网络和上方的依赖安装提示。", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
