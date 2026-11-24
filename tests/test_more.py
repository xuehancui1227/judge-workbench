import io
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from openpyxl import Workbook, load_workbook
from docx import Document

from judgekit.files import load_dataset, export_dataset
from judgekit.references import parse_reference, ReferenceIndex
from judgekit.models import APIConfig, Plan, Concept, Dimension
from judgekit.plans import propose_plan
from judgekit.llm import Client


def test_docx_with_table():
    doc = Document()
    doc.add_paragraph("公司营业时间")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "前台"
    table.cell(0, 1).text = "18点"
    out = io.BytesIO()
    doc.save(out)
    chunks = parse_reference(out.getvalue(), "制度.docx")
    assert any("段落" in c.location for c in chunks)
    assert any("表" in c.location and "18点" in c.text for c in chunks)


def test_json_envelope_preserved():
    source = {"metadata": {"model": "old"}, "records": [{"in": "x", "out": "y"}]}
    ds = load_dataset(json.dumps(source).encode(), "a.json", json_root="records")
    data, _ = export_dataset(ds, {0: {"judge_reason": "ok"}}, "c2")
    result = json.loads(data)
    assert result["metadata"] == source["metadata"]
    assert result["records"][0]["out"] == "y"


def test_bad_csv_and_empty():
    with pytest.raises(ValueError):
        load_dataset(b"a,b\nx,y,z", "a.csv")
    with pytest.raises(ValueError):
        load_dataset(b"a,b\n", "a.csv")


def test_rule_dimensions_not_hardcoded():
    plan = Plan(name="翻译", original_rules="评翻译", evaluation_target="翻译", concepts=[],
                dimensions=[Dimension(key="brief", name="精简程度", description="简练", kind="string", allowed_values=["好", "差"])])
    assert plan.dimensions[0].kind == "string"
    with pytest.raises(ValueError):
        Plan(name="x", original_rules="x", evaluation_target="x", concepts=[],
             dimensions=[Dimension(key="x", name="status", description="reserved")])


def test_xlsx_blank_rows_do_not_shift_results():
    wb = Workbook()
    ws = wb.active
    ws.append(["输入", "输出"])
    ws.append(["a", "b"])
    ws.append([None, None])
    ws.append(["c", "d"])
    out = io.BytesIO()
    wb.save(out)
    ds = load_dataset(out.getvalue(), "a.xlsx")
    result, _ = export_dataset(ds, {1: {"judge_reason": "second"}}, "c2")
    actual = load_workbook(io.BytesIO(result)).active
    assert actual["C4"].value == "second"
    assert actual["C3"].value is None
