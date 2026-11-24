from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from judgekit.llm import CallFailure, Client
from judgekit.models import APIConfig


class FakeHTTP:
    def __init__(self, responses):
        self.responses = responses

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        return self.responses.pop(0)


def response(content, finish="stop"):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3},
    })


@pytest.mark.parametrize(
    "content,finish,reason",
    [
        ("这不是JSON", "stop", "返回内容不是有效JSON"),
        ('{"value":', "length", "输出上限而截断"),
        ('{"value":"wrong"}', "stop", "JSON结构不合规"),
    ],
)
def test_format_failure_has_specific_reason_and_raw_content(monkeypatch, content, finish, reason):
    fake = FakeHTTP([response(content, finish)])
    monkeypatch.setattr("judgekit.llm.httpx.Client", lambda **kwargs: fake)
    cfg = APIConfig(model="test-model", api_key="secret", retries=0, mock=False)
    with pytest.raises(CallFailure) as caught:
        Client(cfg).complete("system", {"sample": 1}, {
            "type": "object", "properties": {"value": {"type": "integer"}},
            "required": ["value"], "additionalProperties": False,
        })
    error = caught.value
    assert error.kind == "format_error"
    assert reason in str(error)
    assert error.details[0]["assistant_content"] == content
    assert error.details[0]["attempt"] == 1


def test_all_failed_attempts_are_available(monkeypatch):
    fake = FakeHTTP([response("bad one"), response("bad two")])
    monkeypatch.setattr("judgekit.llm.httpx.Client", lambda **kwargs: fake)
    cfg = APIConfig(model="test-model", api_key="secret", retries=1, mock=False)
    with pytest.raises(CallFailure) as caught:
        Client(cfg).complete("system", {}, {"type": "object"})
    assert [x["assistant_content"] for x in caught.value.details] == ["bad one", "bad two"]


def test_plan_failure_is_visible_and_downloadable(tmp_path, monkeypatch):
    import judgekit.plans

    details = [{"attempt": 1, "category": "format_error", "reason": "返回内容不是有效JSON",
                "http_status": 200, "finish_reason": "stop", "assistant_content": "raw model answer"}]

    def fail(*args, **kwargs):
        raise CallFailure("模型第1次返回无法使用：返回内容不是有效JSON", kind="format_error", details=details)

    monkeypatch.setattr(judgekit.plans, "propose_plan", fail)
    monkeypatch.setenv("JUDGE_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py"), default_timeout=30).run()
    next(c for c in app.checkbox if c.label == "加载内置演示数据").check().run()
    next(b for b in app.button if b.label == "理解规则并推荐字段").click().run()
    assert not app.exception
    assert any("返回内容不是有效JSON" in x.value for x in app.error)
    downloads = [x for x in app.get("download_button") if x.label == "下载失败诊断（JSON）"]
    assert len(downloads) == 1
    assert "raw model answer" in str(app.session_state["last_plan_diagnostic"])
    assert "api_key" not in str(app.session_state["last_plan_diagnostic"])
