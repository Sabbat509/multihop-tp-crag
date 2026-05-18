#!/usr/bin/env python3
"""Build normalized document data for the MultiHop-RAG corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from naive_rag.index_io import save_pickle


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-file", type=Path, default=Path("datasets/multihop_rag/raw/corpus.jsonl"))
    parser.add_argument("--out-file", type=Path, default=Path("datasets/multihop_rag/intermediate/doc_data.pkl"))
    args = parser.parse_args()

    rows = read_jsonl(args.corpus_file)
    doc_data = []
    for index, row in enumerate(rows):
        title = str(row.get("title") or f"document-{index}").strip()
        text_parts = [
            f"Title: {title}",
            f"Source: {row.get('source') or ''}",
            f"Author: {row.get('author') or ''}",
            f"Published at: {row.get('published_at') or ''}",
            f"Category: {row.get('category') or ''}",
            str(row.get("body") or ""),
        ]
        doc_data.append(
            {
                "id": row.get("url") or f"multihop-{index}",
                "title": title,
                "text": "\n".join(part for part in text_parts if part is not None),
                "source_file": str(args.corpus_file),
                "url": row.get("url"),
                "source": row.get("source"),
                "author": row.get("author"),
                "published_at": row.get("published_at"),
                "category": row.get("category"),
            }
        )
    save_pickle(args.out_file, doc_data)
    print(f"Loaded corpus rows: {len(rows)}")
    print(f"Saved doc_data: {args.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
