from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


class Store:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def path(self, job_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise ValueError("无效运行编号")
        return self.root / job_id

    def create(self, dataset, plan, documents, settings, selected, anchor):
        job_id = uuid.uuid4().hex
        path = self.path(job_id)
        path.mkdir(mode=0o700)
        (path / "source.bin").write_bytes(dataset.content)
        meta = {"id": job_id, "filename": dataset.filename, "sheet": dataset.sheet,
                "header_row": dataset.header_row, "json_root": dataset.json_root,
                "dataset_hash": dataset.fingerprint, "plan": plan.model_dump(),
                "references": documents, "settings": settings, "selected": selected,
                "anchor": anchor, "created": time.time()}
        meta["execution_hash"] = digest({"dataset": dataset.fingerprint, "plan": plan.model_dump(),
                                         "references": documents, "settings": settings})
        self.write_meta(job_id, meta)
        with self.connect(job_id) as db:
            db.execute("CREATE TABLE results (row_index INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
        return job_id

    def connect(self, job_id):
        db = sqlite3.connect(self.path(job_id)/"results.sqlite", timeout=30)
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def write_meta(self, job_id, value):
        path = self.path(job_id)
        temp = path / "meta.tmp"
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        os.replace(temp, path / "meta.json")

    def meta(self, job_id):
        return json.loads((self.path(job_id)/"meta.json").read_text(encoding="utf-8"))

    def list(self):
        out = []
        for path in self.root.glob("*/meta.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                out.append({"id": data["id"], "filename": data["filename"], "created": data["created"],
                            "name": data["plan"]["name"]})
            except (ValueError, KeyError, OSError):
                continue
        return sorted(out, key=lambda x: -x["created"])

    def save_result(self, job_id, index: int, payload):
        with self.connect(job_id) as db:
            db.execute("INSERT OR REPLACE INTO results VALUES (?,?)",
                       (index, json.dumps(payload, ensure_ascii=False, default=str)))

    def results(self, job_id):
        with self.connect(job_id) as db:
            return {i: json.loads(p) for i, p in db.execute("SELECT row_index,payload FROM results ORDER BY row_index")}
