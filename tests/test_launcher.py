import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_batch_windows_format():
    raw = (ROOT/"start.bat").read_bytes()
    assert raw.isascii(), "Keep the cmd bootstrap ASCII-only to avoid codepage issues"
    assert b"\r\n" in raw
    assert b"\n" not in raw.replace(b"\r\n", b"")


@pytest.mark.skipif(os.name != "nt", reason="Windows cmd regression")
def test_double_click_equivalent():
    completed = subprocess.run(["cmd.exe", "/d", "/c", "start.bat --check <NUL"],
                               cwd=ROOT, capture_output=True, timeout=90)
    assert completed.returncode == 0, completed.stdout+completed.stderr
    assert b"JUDGE_LAUNCHER_OK" in completed.stdout
    assert b"can't open file" not in completed.stderr
