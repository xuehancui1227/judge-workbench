from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from copy import copy, deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

MAX_BYTES = 30 * 1024 * 1024


def check_file(data: bytes, name: str):
    if len(data) > MAX_BYTES:
        raise ValueError("单文件不能超过30MB")
    if Path(name).suffix.lower() in {".xlsx", ".docx"}:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if sum(x.file_size for x in z.infolist()) > 150 * 1024 * 1024:
                raise ValueError("文件解压后过大，请拆分文件")


def text_decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise ValueError("文本编码不支持，请另存为UTF-8")


def display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


@dataclass
class Dataset:
    filename: str
    content: bytes
    sheet: str
    header_row: int
    labels: list[str]
    rows: list[dict]
    row_numbers: list[int]
    json_root: str = ""

    @property
    def columns(self):
        return {f"c{i+1}": label for i, label in enumerate(self.labels)}

    @property
    def fingerprint(self):
        return hashlib.sha256(self.content + f"|{self.sheet}|{self.header_row}|{self.json_root}".encode()).hexdigest()

    def profile(self):
        return [{"id": key, "name": label,
                 "type": sorted({type(r.get(key)).__name__ for r in self.rows[:50]}),
                 "examples": [display(r.get(key))[:1200] for r in self.rows[:5]]}
                for key, label in self.columns.items()]


def workbook_sheets(data: bytes) -> list[str]:
    return load_workbook(io.BytesIO(data), read_only=True, data_only=False).sheetnames


def load_dataset(data: bytes, filename: str, sheet: str = "", header_row: int = 1,
                 json_root: str = "") -> Dataset:
    check_file(data, filename)
    ext = Path(filename).suffix.lower()
    indices = []
    if ext == ".xlsx":
        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=False)
        ws = wb[sheet or wb.sheetnames[0]]
        if ws.max_column > 1000 or ws.max_row > 100001:
            raise ValueError("第一版支持最多1000列、100000行，请先缩小数据范围")
        sheet = ws.title
        grid = list(ws.iter_rows(values_only=True))
        if not grid or header_row < 1 or header_row > len(grid):
            raise ValueError("表头行不存在")
        labels = [display(v) or f"未命名列{i+1}" for i, v in enumerate(grid[header_row-1])]
        values = []
        for n, row in enumerate(grid[header_row:], header_row+1):
            if any(v is not None for v in row):
                indices.append(n)
                values.append(row)
        wb.close()
    elif ext == ".csv":
        grid = list(csv.reader(io.StringIO(text_decode(data))))
        if not grid or not 1 <= header_row <= len(grid):
            raise ValueError("CSV为空或表头行不存在")
        labels = grid[header_row-1]
        if any(len(r) > len(labels) for r in grid[header_row:]):
            raise ValueError("CSV存在超出表头的字段，请修正分隔符或表头")
        pairs = [(i, r) for i, r in enumerate(grid[header_row:], header_row+1) if any(r)]
        indices, values = [p[0] for p in pairs], [p[1] for p in pairs]
    elif ext in {".json", ".jsonl"}:
        raw = ([json.loads(line) for line in text_decode(data).splitlines() if line.strip()]
               if ext == ".jsonl" else json.loads(text_decode(data)))
        records = raw.get(json_root) if json_root and isinstance(raw, dict) else raw
        if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
            raise ValueError("JSON需为对象数组，或指定包含对象数组的顶层键；JSONL每行需为对象")
        labels = list(dict.fromkeys(k for r in records for k in r))
        values = [[r.get(k) for k in labels] for r in records]
        indices = list(range(1, len(values)+1))
    else:
        raise ValueError("提测文件支持 xlsx / csv / json / jsonl")
    if not labels or not values:
        raise ValueError("没有可评测的数据行")
    if len(values) > 100000:
        raise ValueError("最多支持100000行")
    rows = [{f"c{i+1}": r[i] if i < len(r) else None for i in range(len(labels))} for r in values]
    return Dataset(filename, data, sheet, header_row, labels, rows, indices, json_root)


def safe_cell(v):
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, default=str)
    return v


def export_dataset(ds: Dataset, results: dict[int, dict], anchor: str, suffix: str = "",
                   only_rows: list[int] | None = None, collision: str = "new") -> tuple[bytes, str]:
    """Insert beside selected column in a COPY. Never mutate uploaded bytes.

    Deliberately reject middle insertion into complex workbooks rather than
    silently breaking formulas, names, tables or drawing anchors.
    """
    keys = list(dict.fromkeys(k for result in results.values() for k in result))
    if not keys:
        raise ValueError("没有可导出的结果")
    names = [k + suffix for k in keys]
    if len(set(names)) != len(names):
        raise ValueError("结果列名不唯一")
    position = list(ds.columns).index(anchor)+1
    collisions = set(names) & set(ds.labels)
    if any(ds.labels.count(n) > 1 for n in collisions):
        raise ValueError("已有评测列重名，无法安全选择覆盖对象；请使用全新版本后缀")
    if collisions and collision != "overwrite":
        raise ValueError("结果列重名，请换版本后缀或选择覆盖已有评测列：" + ", ".join(collisions))
    if collisions and any(not name.startswith("judge_") for name in collisions):
        raise ValueError("禁止覆盖非judge_列")
    ext = Path(ds.filename).suffix.lower()
    if only_rows is not None:
        # Filter exports are intentionally new CSVs; full export preserves source workbook.
        out = io.StringIO(newline="")
        writer = csv.writer(out)
        writer.writerow([*ds.labels, *names])
        for i in only_rows:
            cells = [*[display(v) for v in ds.rows[i].values()],
                     *[display(results.get(i, {}).get(k)) for k in keys]]
            writer.writerow(["'"+v if v.startswith(("=", "+", "-", "@", "\t", "\r")) else v for v in cells])
        return out.getvalue().encode("utf-8-sig"), "待复核与异常.csv"
    if ext == ".xlsx":
        wb = load_workbook(io.BytesIO(ds.content), data_only=False)
        ws = wb[ds.sheet]
        new_names = [n for n in names if n not in ds.labels]
        middle = position < len(ds.labels) and bool(new_names)
        if middle:
            has_formulas = any(c.data_type == "f" for s in wb for row in s for c in row)
            complex_objects = (bool(wb.defined_names) or bool(wb._external_links) or any(
                s.tables or s._charts or s._images or s._pivots or s.auto_filter.ref or
                len(s.conditional_formatting) or len(s.data_validations.dataValidation)
                for s in wb))
            if has_formulas or complex_objects or ws.merged_cells.ranges:
                raise ValueError("原工作簿含公式、合并区域或复杂对象，中间插列可能损坏引用。请将插入位置改为最右列后再导出；原文件未修改。")
            # openpyxl doesn't move widths with insert_cols.
            widths = {i: copy(ws.column_dimensions[ws.cell(1, i).column_letter])
                      for i in range(position+1, ws.max_column+1)}
            ws.insert_cols(position+1, len(new_names))
            for old_i, dimension in sorted(widths.items(), reverse=True):
                letter = ws.cell(1, old_i+len(new_names)).column_letter
                dimension.index = letter
                ws.column_dimensions[letter] = dimension
        destinations = {}
        for name in names:
            if name in ds.labels:
                old = ds.labels.index(name)+1
                destinations[name] = old + (len(new_names) if middle and old > position else 0)
            else:
                destinations[name] = position+new_names.index(name)+1
        for name, col in destinations.items():
            c = ws.cell(ds.header_row, col, name)
            c.font = Font(bold=True, color="243B53")
            c.fill = PatternFill("solid", fgColor="EAF0F5")
            c.alignment = Alignment(wrap_text=True, vertical="top")
            ws.column_dimensions[c.column_letter].width = 24 if "reason" not in name else 48
        for i, result in results.items():
            for key, name in zip(keys, names):
                value = safe_cell(result.get(key))
                if isinstance(value, str) and len(value) > 32767:
                    value = value[:32650]+"\n[超出Excel单元格限制，完整内容见运行详情/JSON日志]"
                c = ws.cell(ds.row_numbers[i], destinations[name], value)
                if isinstance(value, str):
                    c.data_type = "s"  # untrusted model text must never become Excel formulas
                c.alignment = Alignment(wrap_text=True, vertical="top")
                if key == "judge_overall_pass" and isinstance(value, bool):
                    c.fill = PatternFill("solid", fgColor="E8F1EB" if value else "F5E9E5")
                if key == "judge_needs_review" and value is True:
                    c.fill = PatternFill("solid", fgColor="F7F0D9")
        out = io.BytesIO()
        wb.save(out)
        return out.getvalue(), Path(ds.filename).stem+"_评测结果.xlsx"
    if ext == ".csv":
        grid = list(csv.reader(io.StringIO(text_decode(ds.content))))
        new_names = [n for n in names if n not in ds.labels]
        for n, row in enumerate(grid, 1):
            if n < ds.header_row:
                continue
            row.extend([""]*(len(ds.labels)-len(row)))
            row[position:position] = new_names if n == ds.header_row else [""]*len(new_names)
        labels = grid[ds.header_row-1]
        for i, result in results.items():
            for k, name in zip(keys, names):
                value = display(result.get(k))
                # spreadsheet software interprets formula-like CSV cells on opening
                if value.startswith(("=", "+", "-", "@", "\t", "\r")):
                    value = "'"+value
                grid[ds.row_numbers[i]-1][labels.index(name)] = value
        out = io.StringIO(newline="")
        csv.writer(out).writerows(grid)
        return out.getvalue().encode("utf-8-sig"), Path(ds.filename).stem+"_评测结果.csv"
    raw = ([json.loads(line) for line in text_decode(ds.content).splitlines() if line.strip()]
           if ext == ".jsonl" else json.loads(text_decode(ds.content)))
    records = raw[ds.json_root] if ds.json_root else raw
    anchor_name = ds.columns[anchor]
    for i, result in results.items():
        extra = dict(zip(names, (result.get(k) for k in keys)))
        record = records[i]
        updated = {}
        for k, v in record.items():
            updated[k] = extra.pop(k, v)
            if k == anchor_name:
                updated.update({n: extra.pop(n) for n in list(extra) if n not in record})
        updated.update(extra)
        records[i] = updated
    output = ("\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in records)
              if ext == ".jsonl" else json.dumps(raw, ensure_ascii=False, indent=2, default=str))
    return output.encode("utf-8"), Path(ds.filename).stem+"_评测结果"+ext
