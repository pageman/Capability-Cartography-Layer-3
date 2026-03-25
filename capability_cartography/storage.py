"""Persistent storage for sweep results and run registries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence


class RunStorage:
    """Store per-run artifacts plus a tabular sweep registry."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_json(self, relative_path: str | Path, payload: Dict[str, object]) -> str:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        return str(path)

    def append_jsonl(self, relative_path: str | Path, payload: Dict[str, object]) -> str:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        return str(path)

    def save_records_csv(self, relative_path: str | Path, records: Sequence[Dict[str, object]]) -> str:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not records:
            path.write_text("")
            return str(path)
        headers = sorted({key for record in records for key in record.keys()})
        lines = [",".join(headers)]
        for record in records:
            row = []
            for header in headers:
                value = record.get(header, "")
                text = str(value).replace('"', '""')
                if "," in text or '"' in text:
                    text = f'"{text}"'
                row.append(text)
            lines.append(",".join(row))
        path.write_text("\n".join(lines) + "\n")
        return str(path)
