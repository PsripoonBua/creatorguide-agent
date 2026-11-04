"""Load the YouTube Help articles and the chunk index."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class Corpus:
    """Documents (whole articles) keyed by id, with section splitting for read_doc."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = {d["id"]: d for d in docs}
        if len(self.docs) != len(docs):
            raise ValueError("duplicate document ids")

    @classmethod
    def load(cls, path: Path) -> "Corpus":
        return cls(read_jsonl(path))

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.docs

    def title(self, doc_id: str) -> str:
        return self.docs[doc_id]["title"]

    def url(self, doc_id: str) -> str:
        return self.docs[doc_id]["url"]

    def sections(self, doc_id: str) -> list[tuple[str, str]]:
        """Split at '## ' headings (same rule as the index builder)."""
        out: list[tuple[str, str]] = []
        heading, lines = "Introduction", []
        for line in self.docs[doc_id]["text"].splitlines():
            if line.startswith("## "):
                body = "\n".join(lines).strip()
                if body:
                    out.append((heading, body))
                heading, lines = line[3:].strip() or "Untitled section", []
            else:
                lines.append(line)
        body = "\n".join(lines).strip()
        if body:
            out.append((heading, body))
        return out


def doc_id_of(identifier: str) -> str:
    """'yt-72851:0003' -> 'yt-72851'; a bare doc id is returned unchanged."""
    return identifier.split(":", 1)[0]
