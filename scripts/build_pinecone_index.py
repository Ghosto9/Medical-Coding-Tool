#!/usr/bin/env python3
"""Build a Pinecone ICD-10-CM vector index from official CDC/NCHS FY2026 files."""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import urllib.request
import zipfile
from typing import Iterable

SOURCE_URL = (
    "https://ftp.cdc.gov/pub/health_statistics/nchs/publications/"
    "ICD10CM/2026/icd10cm-code-descriptions-2026.zip"
)
DEFAULT_INDEX = "amaze-icd10cm-2026"
DEFAULT_NAMESPACE = "icd10cm_2026"
DEFAULT_MODEL = "text-embedding-3-small"

CODE_RE = re.compile(
    r"^([A-Z][0-9A-Z]{2}(?:\.[0-9A-Z]{1,4})?)\s*"
    r"(?:\||\t|,| {2,})\s*(.+?)\s*$"
)
CODE_ONLY_RE = re.compile(r"^[A-Z][0-9A-Z]{2}(?:\.[0-9A-Z]{1,4})?$")


def download_source() -> bytes:
    req = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "Amaze-Medical-Coding-Tool/1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def extract_description_text(blob: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        candidates = [
            n for n in z.namelist()
            if not n.endswith("/") and n.lower().endswith((".txt", ".csv"))
        ]
        if not candidates:
            raise RuntimeError(
                f"No TXT/CSV file found in source ZIP: {z.namelist()}"
            )
        name = max(candidates, key=lambda n: z.getinfo(n).file_size)
        raw = z.read(name)
    return raw.decode("utf-8-sig", errors="replace")


def parse_catalog(text: str) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        m = CODE_RE.match(line)
        if m:
            code, title = m.group(1), m.group(2).strip()
        else:
            m2 = re.match(
                r"^([A-Z][0-9A-Z]{2}(?:\.[0-9A-Z]{1,4})?)\s+(.+)$",
                line,
            )
            if not m2:
                continue
            code, title = m2.group(1), m2.group(2).strip()

        if not CODE_ONLY_RE.match(code) or code in seen:
            continue
        if title.lower() in {"description", "title"}:
            continue

        seen.add(code)
        rows.append({"code": code, "title": title})

    if not rows:
        raise RuntimeError(
            "Could not parse any ICD-10-CM codes from the official description file."
        )
    return rows


def add_hierarchy(rows: list[dict]) -> list[dict]:
    codes = {r["code"] for r in rows}
    by_code = {r["code"]: r for r in rows}

    def parent_for(code: str) -> str | None:
        candidates = []
        if "." in code:
            left, right = code.split(".", 1)
            if len(right) > 1:
                candidates.append(left + "." + right[:-1])
            candidates.append(left)
        else:
            candidates.append(code[:-1])

        for candidate in candidates:
            if candidate in codes:
                return candidate
        return None

    for row in rows:
        parent = parent_for(row["code"])
        row["parent_code"] = parent
        row["parent_title"] = by_code[parent]["title"] if parent else None
        row["level"] = len(row["code"].replace(".", "")) - 2

    parent_codes = set()
    for code in codes:
        if "." in code:
            left, right = code.split(".", 1)
            for i in range(1, len(right)):
                parent_codes.add(left + "." + right[:i])
            parent_codes.add(left)
        else:
            for i in range(3, len(code)):
                parent_codes.add(code[:i])

    for row in rows:
        row["billable"] = row["code"] not in parent_codes
        row["search_text"] = " | ".join(
            x for x in [row["code"], row["title"], row.get("parent_title")] if x
        )

    return rows


def batched(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def build_pinecone(
    rows: list[dict],
    index_name: str,
    namespace: str,
    model: str,
    batch_size: int,
) -> None:
    from openai import OpenAI
    from pinecone import Pinecone, ServerlessSpec

    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    existing = {x["name"] for x in pc.list_indexes()}
    if index_name not in existing:
        print(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1"),
            ),
        )

    index = pc.Index(index_name)

    for batch_no, batch in enumerate(batched(rows, batch_size), start=1):
        texts = [row["search_text"] for row in batch]
        response = openai_client.embeddings.create(
            model=model,
            input=texts,
        )

        vectors = []
        for row, embedding in zip(batch, response.data):
            metadata = {
                "code": row["code"],
                "title": row["title"],
                "parent_code": row["parent_code"] or "",
                "level": row["level"],
                "billable": row["billable"],
                "source": "CDC/NCHS ICD-10-CM FY2026",
                "effective_from": "2025-10-01",
                "effective_to": "2026-09-30",
            }
            vectors.append(
                {
                    "id": row["code"].replace(".", ""),
                    "values": embedding.embedding,
                    "metadata": metadata,
                }
            )

        index.upsert(vectors=vectors, namespace=namespace)
        print(f"Uploaded batch {batch_no}: {len(batch)} records")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--index", default=os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX))
    parser.add_argument("--namespace", default=os.getenv("PINECONE_NAMESPACE", DEFAULT_NAMESPACE))
    parser.add_argument("--model", default=os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_MODEL))
    args = parser.parse_args()

    print("Downloading official FY2026 ICD-10-CM descriptions...")
    blob = download_source()
    rows = add_hierarchy(parse_catalog(extract_description_text(blob)))

    if args.limit:
        rows = rows[:args.limit]

    print(f"Parsed {len(rows):,} ICD-10-CM records")
    print("Example:", json.dumps(rows[0], ensure_ascii=False))

    if args.dry_run:
        print("Dry run complete. No Pinecone/OpenAI calls were made.")
        return 0

    missing = [
        name for name in ("OPENAI_API_KEY", "PINECONE_API_KEY")
        if not os.getenv(name)
    ]
    if missing:
        raise SystemExit("Missing environment variables: " + ", ".join(missing))

    build_pinecone(rows, args.index, args.namespace, args.model, args.batch_size)
    print(f"Done. Index={args.index} namespace={args.namespace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
