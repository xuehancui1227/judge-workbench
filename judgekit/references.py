from __future__ import annotations

import hashlib
import io
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .files import check_file, load_dataset, text_decode, display, workbook_sheets


@dataclass
class Chunk:
    id: str
    file: str
    location: str
    text: str
    fields: dict = field(default_factory=dict)


def parse_reference(data: bytes, filename: str, chunk_size: int = 1400) -> list[Chunk]:
    check_file(data, filename)
    ext = Path(filename).suffix.lower()
    units = []
    if ext == ".pdf":
        from pypdf import PdfReader
        pdf = PdfReader(io.BytesIO(data))
        if pdf.is_encrypted:
            raise ValueError("加密PDF请先解密")
        units = [(f"第{i+1}页", page.extract_text() or "", {}) for i, page in enumerate(pdf.pages)]
    elif ext == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        units = [(f"段落{i+1}", p.text, {}) for i, p in enumerate(doc.paragraphs) if p.text.strip()]
        for t, table in enumerate(doc.tables):
            units.extend((f"表{t+1}行{r+1}", " | ".join(c.text for c in row.cells), {})
                         for r, row in enumerate(table.rows))
    elif ext == ".json" and isinstance(json.loads(text_decode(data)), dict):
        raw = json.loads(text_decode(data))
        for key, value in raw.items():
            units.append((f"键 {key}", display({key: value}), {"key": key}))
    elif ext in {".xlsx", ".csv", ".json", ".jsonl"}:
        for sheet in workbook_sheets(data) if ext == ".xlsx" else [""]:
            ds = load_dataset(data, filename, sheet)
            for i, row in enumerate(ds.rows):
                fields = {ds.columns[k]: v for k, v in row.items()}
                units.append((f"{sheet or '数据'} 行{ds.row_numbers[i]}", display(fields), fields))
    elif ext in {".txt", ".md"}:
        lines = text_decode(data).splitlines()
        units = [(f"行{i+1}–{min(i+20, len(lines))}", "\n".join(lines[i:i+20]), {})
                 for i in range(0, len(lines), 20)]
    else:
        raise ValueError("不支持的参考资料格式")
    chunks = []
    digest = hashlib.sha256(data).hexdigest()[:12]
    for location, text, fields in units:
        for start in range(0, len(text), max(1, chunk_size-150)):
            part = text[start:start+chunk_size]
            if part.strip():
                chunks.append(Chunk(f"{digest}-{len(chunks)+1}", filename,
                                    location+f" /片段{start//max(1, chunk_size-150)+1}", part, fields))
    if not chunks:
        raise ValueError("未提取到文本；扫描PDF需先转成可复制文本，本工具不做OCR")
    return chunks


def tokenize(text: str) -> list[str]:
    en = re.findall(r"[a-z0-9_]+", text.lower())
    chinese = re.findall(r"[\u4e00-\u9fff]+", text)
    return en+[s[i:i+2] for s in chinese for i in range(max(1, len(s)-1))]


class ReferenceIndex:
    """Local BM25, no embeddings and no remote data transfer."""

    def __init__(self, documents: list[dict]):
        self.documents = documents

    def retrieve(self, row: dict, query: str, top_k: int = 5,
                 max_chars: int = 10000) -> tuple[list[dict], list[str]]:
        candidates = []
        warnings = []
        for doc in self.documents:
            chunks = doc.get("chunks", [])
            mode = doc.get("mode", "auto")
            if mode == "match":
                source = display(row.get(doc.get("column_id", "")))
                if not source:
                    if doc.get("required"):
                        warnings.append(f"必需参考资料 {doc['name']} 缺少匹配值")
                    continue
                selected = [c for c in chunks if display(c.get("fields", {}).get(doc.get("reference_field", ""))) == source]
                if not selected and doc.get("required"):
                    warnings.append(f"必需参考资料 {doc['name']} 无匹配条目")
                candidates.extend(selected)
            elif mode == "global":
                # Never silently drop mandatory global rules.
                if sum(len(c["text"]) for c in chunks) > max_chars:
                    warnings.append(f"全局资料 {doc['name']} 超出预算，请改自动检索或缩短资料")
                candidates.extend(chunks)
            else:
                counts = [Counter(tokenize(c["text"])) for c in chunks]
                if not counts:
                    continue
                terms = set(tokenize(query))
                avglen = sum(sum(c.values()) for c in counts)/len(counts) or 1
                df = {t: sum(t in c for c in counts) for t in terms}
                scored = []
                for chunk, cnt in zip(chunks, counts):
                    length = sum(cnt.values())
                    score = sum(math.log(1+(len(counts)-df[t]+0.5)/(df[t]+0.5)) *
                                cnt[t]*2.5/(cnt[t]+1.5*(0.25+0.75*length/avglen))
                                for t in terms if cnt[t])
                    if score > 0:
                        scored.append((score, chunk))
                selected = [c for _, c in sorted(scored, key=lambda x: -x[0])[:top_k]]
                if not selected and doc.get("required"):
                    warnings.append(f"必需参考资料 {doc['name']} 没有检索命中")
                candidates.extend(selected)
        unique = list({c["id"]: c for c in candidates}.values())
        selected = []
        length = 0
        for c in unique:
            if length+len(c["text"]) > max_chars:
                warnings.append("参考片段超过总预算；需调整资料范围或提高预算，不能据此推断知识库没有相关事实")
                break
            selected.append(c)
            length += len(c["text"])
        return selected, warnings
