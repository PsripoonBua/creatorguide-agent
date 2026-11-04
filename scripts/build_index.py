"""Rebuild the chunk index + MiniLM embeddings (needs sentence-transformers; downloads the model).

Adapted from the author's earlier NLP course project (see README, "Provenance").
The committed index in data/index was built with this script (224-token chunks, 32 overlap).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
REQUIRED_FIELDS = {"id", "title", "url", "category", "text"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_documents(path: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            doc = json.loads(line)
            missing = REQUIRED_FIELDS - doc.keys()
            if missing:
                raise ValueError(f"Line {line_number} is missing {sorted(missing)}")
            if doc["id"] in ids:
                raise ValueError(f"Duplicate document ID: {doc['id']}")
            if not doc["text"].strip():
                raise ValueError(f"Empty article: {doc['id']}")
            ids.add(doc["id"])
            documents.append(doc)
    if not documents:
        raise ValueError(f"No documents in {path}")
    return documents


def sections(text: str) -> Iterable[tuple[str, str]]:
    """Split only at source headings; keep the introductory text as a section."""
    heading = "Introduction"
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            body = "\n".join(lines).strip()
            if body:
                yield heading, body
            heading = line[3:].strip() or "Untitled section"
            lines = []
        else:
            lines.append(line)
    body = "\n".join(lines).strip()
    if body:
        yield heading, body


def clip_text(tokenizer: Any, value: str, token_limit: int) -> str:
    offsets = tokenizer(
        value, add_special_tokens=False, return_offsets_mapping=True, verbose=False
    )["offset_mapping"]
    if len(offsets) <= token_limit:
        return value
    return value[: offsets[token_limit - 1][1]]


def short_prefix(tokenizer: Any, title: str, heading: str) -> str:
    # Reserve most of the model's window for article content.
    short_title = clip_text(tokenizer, title, 40)
    short_heading = clip_text(tokenizer, heading, 40)
    return f"Title: {short_title}\nSection: {short_heading}\n\n"


def make_chunks(
    documents: list[dict[str, Any]],
    tokenizer: Any,
    token_limit: int,
    overlap: int,
) -> list[dict[str, Any]]:
    if token_limit < 128:
        raise ValueError("--chunk-tokens must be at least 128")
    if overlap < 0 or overlap >= token_limit:
        raise ValueError("--overlap must be between 0 and chunk-tokens - 1")

    chunks: list[dict[str, Any]] = []
    for doc in documents:
        number = 0
        for section_index, (heading, body) in enumerate(sections(doc["text"])):
            prefix = short_prefix(tokenizer, doc["title"], heading)
            prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
            special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
            capacity = token_limit - prefix_tokens - special_tokens
            if capacity <= overlap:
                raise ValueError(f"Heading leaves too little room for content: {doc['id']}")

            offsets = tokenizer(
                body, add_special_tokens=False, return_offsets_mapping=True, verbose=False
            )["offset_mapping"]
            start = 0
            part_index = 0
            while start < len(offsets):
                end = min(start + capacity, len(offsets))
                # Preserve the original case, punctuation, bullets, and whitespace.
                # Check the actual string that will be embedded, then shrink.
                while True:
                    piece = body[offsets[start][0] : offsets[end - 1][1]].strip()
                    candidate = prefix + piece
                    actual_tokens = len(
                        tokenizer.encode(candidate, add_special_tokens=True, verbose=False)
                    )
                    if actual_tokens <= token_limit:
                        break
                    end -= 1
                    if end <= start:
                        raise ValueError(f"Cannot fit a token in {doc['id']} / {heading}")
                if not piece:
                    raise ValueError(f"Empty decoded chunk in {doc['id']} / {heading}")

                number += 1
                chunks.append(
                    {
                        "chunk_id": f"{doc['id']}:{number:04d}",
                        "doc_id": doc["id"],
                        "title": doc["title"],
                        "heading": heading,
                        "section_index": section_index,
                        "part_index": part_index,
                        "category": doc["category"],
                        "url": doc["url"],
                        "text": candidate,
                        "token_count": actual_tokens,
                    }
                )
                part_index += 1
                if end == len(offsets):
                    break
                # If shrinking left the chunk no longer than the overlap, skip the
                # overlap: start must never move backwards.
                start = end - overlap if end - overlap > start else end
    return chunks


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


CHUNK_COLUMNS = [
    "chunk_id", "doc_id", "title", "heading", "section_index", "part_index",
    "category", "url", "token_count", "text",
]


def export_readable_files(output_path: Path) -> None:
    """Create spreadsheet-friendly copies without downloading the model again."""
    import numpy as np

    manifest_path = output_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with (output_path / "chunks.jsonl").open(encoding="utf-8") as stream:
        chunks = [json.loads(line) for line in stream]
    embeddings = np.load(output_path / "embeddings.npy")
    if len(chunks) != manifest["chunks"] or embeddings.shape != (
        len(chunks), manifest["dimensions"]
    ):
        raise ValueError("Index files do not match the manifest")
    if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk_id in chunks.jsonl")

    with (output_path / "chunks.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row_index", *CHUNK_COLUMNS])
        for row_index, chunk in enumerate(chunks):
            writer.writerow([row_index, *(chunk[column] for column in CHUNK_COLUMNS)])

    dimensions = manifest["dimensions"]
    with (output_path / "embeddings.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["row_index", "chunk_id", *(f"dim_{i:03d}" for i in range(dimensions))]
        )
        for row_index, (chunk, vector) in enumerate(zip(chunks, embeddings, strict=True)):
            writer.writerow(
                [row_index, chunk["chunk_id"], *(format(float(value), ".9g") for value in vector)]
            )

    manifest["format_version"] = 2
    manifest["row_alignment"] = (
        "row_index 0..chunks-1 and chunk_id match across chunks.jsonl, chunks.csv, "
        "embeddings.npy, and embeddings.csv"
    )
    manifest["files"] = {
        "chunks.jsonl": "Canonical UTF-8 chunk records, one JSON object per line",
        "chunks.csv": "Readable UTF-8 CSV copy of all chunks, including full text",
        "embeddings.npy": "Canonical float32 matrix for fast retrieval",
        "embeddings.csv": "Readable CSV copy; row_index and chunk_id precede dimensions",
        "manifest.json": "Machine-readable index metadata",
        "manifest.md": "Human-readable index metadata and handoff instructions",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown = f"""# Dense Retrieval Index — อ่านและส่งต่องาน

ชุดนี้สร้างจาก YouTube Help **{manifest['documents']} เอกสาร** เป็น **{manifest['chunks']} chunks**. แต่ละ embedding มี **{dimensions} ค่า** และใช้โมเดล `{manifest['model']}` บน CPU.

## เปิดไฟล์ไหน

| งาน | ไฟล์ |
|---|---|
| เปิดดูข้อความ chunk ใน Excel/Google Sheets | `chunks.csv` |
| เปิดดูค่าตัวเลข embedding ใน Excel/Google Sheets | `embeddings.csv` |
| อ่าน metadata และการตั้งค่า | `manifest.md` หรือ `manifest.json` |
| อ่านด้วย Python/ทำ BM25 | `chunks.jsonl` |
| ค้นหา dense แบบเร็ว | `embeddings.npy` |

ไฟล์ CSV ใช้ **UTF-8 with BOM** เพื่อให้ Excel แสดงตัวอักษรได้ถูกต้อง. `chunks.csv` มีข้อความเต็มในคอลัมน์ `text`; แต่ละแถวคือหนึ่ง chunk. `embeddings.csv` มี {dimensions} คอลัมน์ `dim_000` ถึง `dim_{dimensions - 1:03d}` หลัง `row_index` และ `chunk_id`.

## จับคู่และทำงานต่อ

`row_index` เริ่มที่ 0 และตรงกับลำดับบรรทัดใน `chunks.jsonl` รวมถึงแถวใน `embeddings.npy`. ให้ใช้ `chunk_id` เป็นคีย์เวลา join กับ BM25 หรือผลจากระบบอื่น. **อย่าเรียงไฟล์ CSV ใหม่แล้วสมมติว่า row_index เปลี่ยนตาม**; คอลัมน์นี้ระบุตำแหน่งใน index เดิม.

ถ้าแก้ source หรือเปลี่ยนวิธี chunking ให้รัน `python dense_retrieval.py build` เพื่อสร้างทุกไฟล์ใหม่ให้ตรงกัน. ถ้าต้องการสร้างแค่ไฟล์ CSV/Markdown จาก index เดิม ให้รัน `python dense_retrieval.py export-readable` (ไม่ต้องดาวน์โหลดโมเดล).

## การตั้งค่าของ index นี้

| รายการ | ค่า |
|---|---|
| โมเดล | `{manifest['model']}` |
| revision | `{manifest['model_revision']}` |
| จำนวนเอกสาร | {manifest['documents']} |
| จำนวน chunks | {manifest['chunks']} |
| มิติเวกเตอร์ | {dimensions} |
| token สูงสุดต่อ chunk | {manifest['chunk_tokens']} |
| overlap ระหว่าง chunks | {manifest['overlap_tokens']} tokens |
| embedding normalization | {manifest['embedding_normalization']} |
| การให้คะแนน | {manifest['scoring']} |
| SHA-256 ของ corpus | `{manifest['source_sha256']}` |

ไฟล์ต้นฉบับอยู่ที่ `{manifest['source_file']}` ภายในโฟลเดอร์ `2.Dense_Retrieval`. ดูคำสั่งติดตั้งและตัวอย่างค้นหาใน `../README.md`.
"""
    (output_path / "manifest.md").write_text(markdown, encoding="utf-8")
    print(f"Exported readable CSV and Markdown files in {output_path}")


def build(args: argparse.Namespace) -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    docs = read_documents(input_path)
    model = SentenceTransformer(args.model, device="cpu", revision=args.model_revision)
    token_limit = min(args.chunk_tokens, model.max_seq_length)
    if token_limit != args.chunk_tokens:
        print(f"Model maximum is {model.max_seq_length}; using {token_limit} tokens.")
    chunks = make_chunks(docs, model.tokenizer, token_limit, args.overlap)
    if {chunk["doc_id"] for chunk in chunks} != {doc["id"] for doc in docs}:
        raise ValueError("At least one source document produced no chunks")

    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError("Embedding count differs from chunk count")

    output_path.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path / "chunks.jsonl", chunks)
    np.save(output_path / "embeddings.npy", embeddings)
    manifest = {
        "model": args.model,
        "model_revision": args.model_revision,
        "device": "cpu",
        "source_file": str(input_path.relative_to(ROOT)) if input_path.is_relative_to(ROOT) else str(input_path),
        "source_sha256": file_sha256(input_path),
        "documents": len(docs),
        "chunks": len(chunks),
        "dimensions": embeddings.shape[1],
        "chunk_tokens": token_limit,
        "overlap_tokens": args.overlap,
        "embedding_normalization": "L2",
        "scoring": "dot product (cosine for L2-normalized vectors)",
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    export_readable_files(output_path)
    print(f"Built {len(chunks)} chunks from {len(docs)} documents in {output_path}")


def export_readable(args: argparse.Namespace) -> None:
    export_readable_files(args.output.resolve())


class DenseRetriever:
    """Load the shared chunk index once for use in a demo or hybrid retriever."""

    def __init__(self, index_dir: Path = ROOT / "data" / "index") -> None:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        output_path = Path(index_dir).resolve()
        self.manifest = json.loads(
            (output_path / "manifest.json").read_text(encoding="utf-8")
        )
        with (output_path / "chunks.jsonl").open(encoding="utf-8") as stream:
            self.chunks = [json.loads(line) for line in stream]
        self.embeddings = np.load(output_path / "embeddings.npy")
        if self.embeddings.shape != (len(self.chunks), self.manifest["dimensions"]):
            raise ValueError("Index files have inconsistent dimensions")
        source_path = Path(self.manifest["source_file"])
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        if source_path.exists() and file_sha256(source_path) != self.manifest["source_sha256"]:
            raise ValueError("Source corpus changed. Rebuild the index before searching.")
        self.model = SentenceTransformer(
            self.manifest["model"],
            device="cpu",
            revision=self.manifest["model_revision"],
        )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        import numpy as np

        if not query.strip():
            raise ValueError("Query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        vector = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        scores = self.embeddings @ np.asarray(vector, dtype=np.float32)
        ranked = np.argsort(-scores, kind="stable")[: min(top_k, len(self.chunks))]
        return [
            {"rank": rank, "score": round(float(scores[i]), 6), **self.chunks[i]}
            for rank, i in enumerate(ranked, 1)
        ]


def search(args: argparse.Namespace) -> None:
    results = DenseRetriever(args.output).search(args.query, args.top_k)
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for hit in results:
            print(f"{hit['rank']}. {hit['title']} — {hit['heading']}  score={hit['score']:.4f}")
            print(f"   {hit['chunk_id']}  {hit['url']}")
            print(f"   {hit['text'][:350].replace(chr(10), ' ')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="Create chunks and CPU embeddings")
    build_parser.add_argument("--input", type=Path, default=ROOT / "data" / "docs.jsonl")
    build_parser.add_argument("--output", type=Path, default=ROOT / "data" / "index")
    build_parser.add_argument("--model", default=DEFAULT_MODEL)
    build_parser.add_argument("--model-revision", default=DEFAULT_REVISION)
    build_parser.add_argument("--chunk-tokens", type=int, default=224)
    build_parser.add_argument("--overlap", type=int, default=32)
    build_parser.add_argument("--batch-size", type=int, default=32)
    build_parser.set_defaults(func=build)

    search_parser = commands.add_parser("search", help="Search indexed YouTube articles")
    search_parser.add_argument("query")
    search_parser.add_argument("--output", type=Path, default=ROOT / "data" / "index")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--json", action="store_true")
    search_parser.set_defaults(func=search)

    export_parser = commands.add_parser(
        "export-readable", help="Create CSV and Markdown copies from an existing index"
    )
    export_parser.add_argument("--output", type=Path, default=ROOT / "data" / "index")
    export_parser.set_defaults(func=export_readable)

    args = parser.parse_args()
    if args.command == "search" and args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.command == "build" and args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    args.func(args)


if __name__ == "__main__":
    main()
