"""Generate synthetic sample workbook and explicitly-labelled mock results."""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openpyxl import Workbook
from judgekit.files import load_dataset, export_dataset
from judgekit.models import APIConfig
from judgekit.llm import Client
from judgekit.plans import propose_plan
from judgekit.engine import run_one
from judgekit.references import ReferenceIndex

root = Path(__file__).resolve().parents[1]/"examples"
ds = load_dataset((root/"cases.csv").read_bytes(), "cases.csv")
wb = Workbook()
ws = wb.active
ws.title = "提测数据"
ws.append(ds.labels)
for row in ds.rows:
    ws.append(list(row.values()))
wb.create_sheet("说明")["A1"] = "全部为虚构演示数据。模拟评测结果不代表实际模型表现。"
wb.save(root/"cases.xlsx")
p = propose_plan((root/"rules.md").read_text(encoding="utf-8"), ds.profile(), Client(APIConfig(mock=True)))
(root/"demo-plan.json").write_text(p.model_dump_json(indent=2), encoding="utf-8")
ds = load_dataset((root/"cases.xlsx").read_bytes(), "cases.xlsx", "提测数据")
results = {i: run_one(r, p, ReferenceIndex([]), Client(APIConfig(mock=True)), {})["flat"] for i, r in enumerate(ds.rows)}
data, _ = export_dataset(ds, results, "c3")
(root/"demo-result-MOCK.xlsx").write_bytes(data)
print("已生成演示Excel、方案和明确标注MOCK的结果文件")
