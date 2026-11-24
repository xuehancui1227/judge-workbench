import io
import json
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font
from jsonschema import validate, ValidationError

from judgekit.models import APIConfig, Plan, Concept, Dimension, Rule, result_schema
from judgekit.files import load_dataset, export_dataset
from judgekit.references import ReferenceIndex, parse_reference
from judgekit.llm import Client, CallFailure, redact
from judgekit.engine import run_one, Registry
from judgekit.plans import propose_plan, validate_mapping
from judgekit.storage import Store


@pytest.fixture
def plan():
    return Plan(name="测试", evaluation_target="回答", original_rules="遵循事实", concepts=[
        Concept(key="q", description="问题", source="column", column_ids=["c1"]),
        Concept(key="a", description="回答", source="column", column_ids=["c2"])],
        dimensions=[Dimension(key="faith", name="忠实性", description="有无依据", allowed_values=[0, 1, 3, 5])],
        pass_condition="全部维度至少3分，无重大错误")


def csv_ds():
    return load_dataset("问题,回答,备注\n几点下班,18点,原始\n预订,已订,备注".encode(), "a.csv")


def test_duplicate_headers_and_schema(plan):
    ds = load_dataset(b"x,x\na,b\n", "x.csv")
    assert ds.rows[0] == {"c1": "a", "c2": "b"}
    assert ds.labels == ["x", "x"]
    with pytest.raises(ValueError):
        validate_mapping(plan.model_copy(update={"concepts": [Concept(key="x", description="x", source="column", column_ids=["c99"])]}), ds.columns)


def test_dynamic_dimensions(plan):
    r = {"judgeable": True, "overall_pass": True, "reason": "ok", "evidence": [], "error_types": [],
         "critical_error": False, "needs_review": False, "reference_ids": [], "scores": {"faith": 5}}
    validate(r, result_schema(plan))
    r["scores"]["faith"] = 4
    with pytest.raises(ValidationError):
        validate(r, result_schema(plan))


def test_missing_mock_and_reference(plan):
    client = Client(APIConfig(mock=True))
    result = run_one({"c1": "问题", "c2": ""}, plan, ReferenceIndex([]), client, {})
    assert result["flat"]["judge_status"] == "insufficient_evidence"
    assert result["flat"]["judge_overall_pass"] is None
    result = run_one(csv_ds().rows[0], plan, ReferenceIndex([]), client, {})
    assert result["flat"]["judge_status"] == "mock"
    assert result["flat"]["judge_忠实性"] is None
    plan.reference_required = True
    assert run_one(csv_ds().rows[0], plan, ReferenceIndex([]), client, {})["flat"]["judge_status"] == "insufficient_evidence"


def test_bm25_match_and_budget():
    chunks = [asdict(c) for c in parse_reference("产品,内容\nA,苹果手机的使用说明\nB,香蕉的保鲜方法".encode(), "kb.csv")]
    idx = ReferenceIndex([{"name": "kb", "mode": "auto", "chunks": chunks}])
    found, errors = idx.retrieve({}, "香蕉怎么保鲜")
    assert "香蕉" in found[0]["text"] and not errors
    idx = ReferenceIndex([{"name": "kb", "mode": "match", "column_id": "c1", "reference_field": "产品", "required": True, "chunks": chunks}])
    assert idx.retrieve({"c1": "B"}, "")[0][0]["fields"]["产品"] == "B"
    assert idx.retrieve({"c1": "C"}, "")[1]
    idx = ReferenceIndex([{"name": "kb", "mode": "global", "chunks": chunks}])
    assert idx.retrieve({}, "", max_chars=1)[1]


def test_excel_export_preserves_and_inserts():
    wb = Workbook()
    ws = wb.active
    ws.title = "测试"
    ws.append(["问题", "回答", "备注"])
    ws.append(["q", "a", "untouched"])
    ws["C2"].font = Font(bold=True)
    wb.create_sheet("不修改")["A1"] = "其他内容"
    buffer = io.BytesIO()
    wb.save(buffer)
    ds = load_dataset(buffer.getvalue(), "test.xlsx", "测试")
    result, _ = export_dataset(ds, {0: {"judge_reason": "=HYPERLINK(\"bad\")", "judge_overall_pass": True}}, "c2")
    actual = load_workbook(io.BytesIO(result))
    assert actual["测试"]["C1"].value == "judge_reason"
    assert actual["测试"]["E2"].value == "untouched"
    assert actual["测试"]["E2"].font.bold
    assert actual["测试"]["C2"].data_type == "s"
    assert actual["不修改"]["A1"].value == "其他内容"
    assert ds.content == buffer.getvalue()


def test_excel_formula_guard():
    wb = Workbook()
    ws = wb.active
    ws.append(["问题", "回答", "计算"])
    ws.append(["q", "a", "=1+1"])
    b = io.BytesIO()
    wb.save(b)
    ds = load_dataset(b.getvalue(), "a.xlsx")
    with pytest.raises(ValueError, match="最右"):
        export_dataset(ds, {0: {"judge_reason": "ok"}}, "c2")
    data, _ = export_dataset(ds, {0: {"judge_reason": "ok"}}, "c3")
    assert load_workbook(io.BytesIO(data)).active["C2"].value == "=1+1"


@pytest.mark.parametrize("ext", ["json", "jsonl"])
def test_json_roundtrip(ext):
    rows = [{"q": "a", "answer": {"text": "b"}, "extra": [1, 2]}]
    text = json.dumps(rows) if ext == "json" else json.dumps(rows[0])
    ds = load_dataset(text.encode(), "a."+ext)
    output, _ = export_dataset(ds, {0: {"judge_reason": "ok"}}, "c2")
    actual = json.loads(output)
    obj = actual[0] if ext == "json" else actual
    assert obj["extra"] == [1, 2] and obj["answer"] == {"text": "b"}
    assert list(obj) == ["q", "answer", "judge_reason", "extra"]


def test_csv_collision_and_filter():
    ds = csv_ds()
    out, _ = export_dataset(ds, {0: {"judge_reason": "=cmd"}}, "c2")
    assert "'\u003dcmd" in out.decode("utf-8-sig")
    assert out.decode("utf-8-sig").splitlines()[0] == "问题,回答,judge_reason,备注"
    ds2 = load_dataset(out, "b.csv")
    with pytest.raises(ValueError):
        export_dataset(ds2, {0: {"judge_reason": "x"}}, "c2")
    out2, _ = export_dataset(ds2, {0: {"judge_reason": "new"}}, "c2", collision="overwrite")
    assert "new" in out2.decode("utf-8-sig")


def test_worker_persist_resume_and_no_key(tmp_path, plan):
    store = Store(tmp_path)
    ds = csv_ds()
    cfg = APIConfig(mock=True, api_key="sk-test-secret")
    job = store.create(ds, plan, [], {"api": cfg.model_dump()}, [0, 1], "c2")
    registry = Registry()
    w = registry.launch(store, job, cfg)
    w.thread.join(10)
    assert w.state == "finished" and len(store.results(job)) == 2
    before = store.results(job)
    w2 = registry.launch(store, job, cfg)
    w2.thread.join(10)
    assert store.results(job) == before
    assert "sk-test-secret" not in (store.path(job)/"meta.json").read_text(encoding="utf-8")
    assert w.config.api_key == ""
    assert cfg.api_key == "sk-test-secret"


def test_redaction():
    assert redact({"api_key": "secret", "value": "xsecretx"}, "secret") == {"value": "x[REDACTED]x"}


@pytest.fixture
def api_server():
    state = {"count": 0, "invalid_first": False, "status": 200, "bodies": [], "critical": False}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["bodies"].append(body)
            state["count"] += 1
            payload = json.loads(body["messages"][1]["content"])
            score_keys = payload.get("schema", {}).get("properties", {}).get("scores", {}).get("properties", {})
            result = {"judgeable": True, "overall_pass": True, "reason": "符合资料", "evidence": ["已提供证据"],
                      "error_types": [], "critical_error": state["critical"], "needs_review": False,
                      "reference_ids": [], "scores": {key: 5 for key in score_keys}}
            content = "not json" if state["invalid_first"] and state["count"] == 1 else json.dumps(result)
            data = json.dumps({"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 100, "completion_tokens": 30}}).encode()
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield state, f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()
    server.server_close()


def test_api_retry_usage_and_critical_gate(api_server, plan):
    state, url = api_server
    state["invalid_first"] = True
    client = Client(APIConfig(base_url=url, model="test", api_key="test-key", retries=1))
    p = run_one(csv_ds().rows[0], plan, ReferenceIndex([]), client, {})
    assert p["flat"]["judge_status"] == "completed"
    assert p["flat"]["judge_overall_pass"] is True
    assert p["usage"]["input_tokens"] == 200
    state["critical"] = True
    p = run_one(csv_ds().rows[0], plan, ReferenceIndex([]), client, {})
    assert p["flat"]["judge_overall_pass"] is False
    assert p["flat"]["judge_needs_review"] is True


def test_api_error_does_not_leak_key(api_server, plan):
    state, url = api_server
    state["status"] = 401
    secret = "never-write-me"
    client = Client(APIConfig(base_url=url, model="test", api_key=secret))
    p = run_one(csv_ds().rows[0], plan, ReferenceIndex([]), client, {})
    assert p["flat"]["judge_status"] == "api_error"
    assert secret not in json.dumps(p)


def test_rule_and_judge_conflict(api_server, plan):
    state, url = api_server
    plan.rules = [Rule(name="精确答案", concept="a", operator="equals", expected="19点", severity="fail")]
    p = run_one(csv_ds().rows[0], plan, ReferenceIndex([]), Client(APIConfig(base_url=url, model="test", api_key="test")), {})
    assert p["flat"]["judge_overall_pass"] is False
    assert p["flat"]["judge_needs_review"] is True
    assert "rule_judge_conflict" in p["flat"]["judge_error_type"]
