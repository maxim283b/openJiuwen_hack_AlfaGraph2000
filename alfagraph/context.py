"""Bounded task previews and literal retrieval, independent of benchmark templates."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path


def workspace_files(work: Path):
    return sorted(p for p in work.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def task_context(attempt, limit=6500):
    files = workspace_files(attempt.work)
    parts = ["Available files:"]
    used = 0
    # Include small sources/records directly; large tables need schema and samples.
    for path in files[:100]:
        rel = str(path.relative_to(attempt.work))
        header = f"\nFILE {rel} ({path.stat().st_size} bytes)"
        excerpt = ""
        try:
            if path.suffix.lower() in {".py", ".md", ".txt", ".json", ".csv"}:
                with path.open(errors="replace") as stream:
                    text = stream.read(2400)
                if path.suffix.lower() == ".csv":
                    rows = list(csv.reader(io.StringIO(text)))[:4]
                    excerpt = "\nCSV header and up to 3 sample rows: " + json.dumps(rows)
                elif path.stat().st_size <= 1800:
                    excerpt = "\n" + text
                else:
                    excerpt = "\nPrefix (truncated):\n" + text[:700]
        except (OSError, csv.Error):
            pass
        block = header + excerpt
        if used + len(block.encode()) > limit:
            block = header
        if used + len(block.encode()) > limit:
            parts.append("\nMore files omitted; use list_files.")
            break
        used += len(block.encode())
        parts.append(block)
        if excerpt:
            attempt.evidence_files.add(rel)
    if len(files) > 100:
        parts.append("\nMore files omitted; use list_files.")
    if attempt.test_result:
        parts.append("\nInitial test result: " + json.dumps(attempt.test_result)[:1400])
    return "\n".join(parts)


def search_files(work: Path, query: str):
    if not query.strip() or len(query) > 200:
        raise ValueError("Provide a nonempty literal query, at most 200 characters")
    matches, used = [], 0
    for path in workspace_files(work):
        if path.suffix.lower() not in {".md", ".txt", ".json", ".csv", ".py"}:
            continue
        with path.open(errors="replace") as stream:
            lines = stream.read(1_000_000).splitlines()
        for index, line in enumerate(lines):
            if query.casefold() not in line.casefold():
                continue
            snippet = "\n".join(lines[max(0, index - 1) : index + 3])[:900]
            match = {"path": str(path.relative_to(work)), "line": index + 1, "text": snippet}
            used += len(json.dumps(match).encode())
            if len(matches) >= 12 or used > 5000:
                return {"matches": matches, "truncated": True}
            matches.append(match)
    return {"matches": matches, "truncated": False}
