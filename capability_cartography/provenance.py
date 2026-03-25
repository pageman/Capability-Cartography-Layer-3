"""Repository provenance helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict


def _git(path: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def repository_provenance(*, name: str, url: str, root: Path | None) -> Dict[str, Any]:
    exists = bool(root and root.exists())
    commit = _git(root, "rev-parse", "HEAD") if exists else None
    short_commit = _git(root, "rev-parse", "--short", "HEAD") if exists else None
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD") if exists else None
    status = _git(root, "status", "--porcelain") if exists else None
    return {
        "name": name,
        "url": url,
        "configured_root": str(root) if root else None,
        "available": exists,
        "commit": commit,
        "short_commit": short_commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else False,
    }
