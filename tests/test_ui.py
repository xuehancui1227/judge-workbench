from pathlib import Path
import time
import json

from streamlit.testing.v1 import AppTest


def test_ui_demo(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py"), default_timeout=30)
    app.run()
    assert not app.exception
    next(c for c in app.checkbox if c.label == "加载内置演示数据").check().run()
    assert not app.exception
    next(b for b in app.button if b.label == "理解规则并推荐字段").click().run()
    assert not app.exception
    assert any(b.label == "确认此方案" for b in app.button)
    next(b for b in app.button if b.label == "确认此方案").click().run()
    assert not app.exception
    next(c for c in app.checkbox if c.label.startswith("确认：所选证据")).check().run()
    next(b for b in app.button if b.label == "创建并开始评测").click().run()
    assert not app.exception
    assert any("已启动" in x.value for x in app.success)
    from judgekit.storage import Store
    store = Store(tmp_path/"jobs")
    jobs = store.list()
    assert len(jobs) == 1
    limit = time.monotonic()+10
    while len(store.results(jobs[0]["id"])) < 3 and time.monotonic() < limit:
        time.sleep(0.05)
    assert len(store.results(jobs[0]["id"])) == 3
    app.run()
    next(b for b in app.button if b.label == "生成结果文件").click().run()
    assert not app.exception
    assert any(b.label == "下载完整结果文件" for b in app.get("download_button"))
