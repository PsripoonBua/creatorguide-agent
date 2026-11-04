"""Helpers shared by the evaluation scripts."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def meta(**extra: Any) -> dict[str, Any]:
    return {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "questions_sha256": file_sha(ROOT / "eval" / "questions.jsonl"),
            "corpus_sha256": file_sha(ROOT / "data" / "docs.jsonl"), **extra}


def write_json(name: str, obj: Any) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_json(name: str) -> Any | None:
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def try_build(kind: str, settings=None, **kw):
    """Build a retriever; return (retriever, None) or (None, reason) when dense deps are unavailable."""
    from creatorguide.config import ROOT as R
    from creatorguide.retrieval import build_retriever

    embed_backend = getattr(settings, "embed_backend", "local")
    embed_model = getattr(settings, "embed_model", None)
    try:
        return build_retriever(kind, R / "data" / "index", embed_backend=embed_backend,
                               embed_model=embed_model, **kw), None
    except (ImportError, OSError) as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:160]}"
