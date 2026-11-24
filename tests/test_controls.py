import io
import json
import threading
import time
from dataclasses import asdict

import pytest
from openpyxl import Workbook, load_workbook

from judgekit.models import APIConfig, Concept, Dimension, Plan
from judgekit.llm import Client, CallFailure
from judgekit.files import load_dataset, export_dataset
from judgekit.engine import Registry, run_one
from judgekit.references import ReferenceIndex, parse_reference
from judgekit.storage import Store


def test_pause_resume_stop_and_skip(tmp_path, monkeypatch):
    import judgekit.engine as engine
    started = threading.Event()
    release = threading.Event()
    seen = []
    def slow(row, plan, references, client, options, cancel=None):
        seen.append(row["c1"])
        started.set()
        release.wait(5)
        return {"flat": {"judge_status": "mock", "judge_needs_review": True}, "seconds": 0}
    monkeypatch.setattr(engine, "run_one", slow)
    ds = load_dataset(b"q,a\na,x\nb,y\nc,z", "test.csv")
    p = Plan(evaluation_target="x", original_rules="x", concepts=[])
    cfg = APIConfig(mock=True, concurrency=1)
    store = Store(tmp_path)
    job = store.create(ds, p, [], {"api": cfg.model_dump()}, [0, 1, 2], "c2")
    reg = Registry()
    w = reg.launch(store, job, cfg)
    assert started.wait(5)
    w.pause()
    release.set()
    limit = time.monotonic()+5
    while not store.results(job) and time.monotonic() < limit:
        time.sleep(0.01)
    assert len(store.results(job)) == 1
    assert len(seen) == 1
    with pytest.raises(ValueError, match="正在运行"):
        reg.launch(store, job, cfg)
    w.stop()
    w.thread.join(5)
    assert w.state == "stopped"
    reg.launch(store, job, cfg).thread.join(5)
    assert seen == ["a", "b", "c"]
    assert len(store.results(job)) == 3


def test_input_budget_no_api():
    p = Plan(evaluation_target="x", original_rules="x", concepts=[Concept(key="q", description="x", source="column", column_ids=["c1"])])
    result = run_one({"c1": "long"*3000}, p, ReferenceIndex([]), Client(APIConfig(max_chars=1000, api_key="x", model="x")), {})
    assert result["flat"]["judge_status"] == "input_too_long"


def test_overwrite_column_and_insert_new_fields():
    wb = Workbook()
    ws = wb.active
    ws.append(["q", "answer", "judge_reason", "rest"])
    ws.append(["a", "b", "old", "unchanged"])
    b = io.BytesIO()
    wb.save(b)
    ds = load_dataset(b.getvalue(), "a.xlsx")
    output, _ = export_dataset(ds, {0: {"judge_reason": "new", "judge_status": "completed"}}, "c2", collision="overwrite")
    actual = load_workbook(io.BytesIO(output)).active
    assert [c.value for c in actual[1]] == ["q", "answer", "judge_status", "judge_reason", "rest"]
    assert [c.value for c in actual[2]] == ["a", "b", "completed", "new", "unchanged"]


def test_json_reference_object():
    chunks = parse_reference(b'{"policy":"rule text", "faq":[{"q":"hi"}]}', "kb.json")
    assert chunks[0].fields["key"] == "policy"
    assert len(chunks) == 2
