from pathlib import Path

from streamlit.testing.v1 import AppTest

from judgekit.connection import ProbeResult


def test_connection_button_uses_probe_not_batch_client(tmp_path, monkeypatch):
    import judgekit.connection
    seen = []

    def fake_probe(cfg, on_progress):
        seen.append(cfg.model_copy(deep=True))
        on_progress(0.0, "正在发送轻量请求")
        on_progress(0.1, "已收到服务响应 HTTP 200")
        return ProbeResult("ok", "当前模型与输出模式测试通过", 0.1, 0.1, 200)

    monkeypatch.setattr(judgekit.connection, "probe_connection", fake_probe)
    monkeypatch.setenv("JUDGE_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py"), default_timeout=30).run()
    app.toggle[0].set_value(False)
    next(x for x in app.text_input if x.label == "裁判模型名称").set_value("local-test")
    next(x for x in app.text_input if x.label == "API Key").set_value("mock-only-private-key")
    next(x for x in app.number_input if x.label == "单次超时（秒）").set_value(600)
    next(x for x in app.number_input if x.label == "额外重试次数").set_value(5)
    next(x for x in app.number_input if x.label == "最大输出Token").set_value(30000)
    next(x for x in app.button if x.label == "测试连接").click().run()
    assert not app.exception
    assert len(seen) == 1
    assert seen[0].timeout == 600 and seen[0].retries == 5 and seen[0].max_output_tokens == 30000
    assert any("测试通过" in x.value for x in app.success)
    assert next(x for x in app.number_input if x.label == "单次超时（秒）").value == 600
    assert next(x for x in app.number_input if x.label == "额外重试次数").value == 5
