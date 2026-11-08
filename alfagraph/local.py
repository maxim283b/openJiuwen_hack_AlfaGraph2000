"""Conservative parsers: unsupported wording falls back to the agent, never a guess."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from alfagraph.tasks import Task


class InsufficientData(Exception):
    pass


def solve_math(instruction: str) -> str | None:
    body = instruction.split("How many boxes", 1)[0].strip()
    start = re.match(r"A warehouse starts with (\d+) boxes\.\s*", body, re.I)
    if not start:
        return None
    value = int(start[1])
    for clause in re.split(r"\.\s*", body[start.end() :].strip()):
        if not clause:
            continue
        if m := re.fullmatch(r"It ships (\d+) boxes per day for (\d+) days", clause, re.I):
            value -= int(m[1]) * int(m[2])
        elif m := re.fullmatch(r"It receives (\d+) deliveries of (\d+) boxes each", clause, re.I):
            value += int(m[1]) * int(m[2])
        elif m := re.fullmatch(
            r"(?:Then )?the stock is split equally into (\d+) warehouses and it keeps "
            r"one share \(rounded down\)",
            clause,
            re.I,
        ):
            if int(m[1]) == 0:
                raise InsufficientData("Division into zero warehouses")
            value //= int(m[1])
        elif re.fullmatch(
            r"(?:Then )?the stock is doubled by merging with a sister warehouse", clause, re.I
        ):
            value *= 2
        elif m := re.fullmatch(r"It (receives|ships|loses) (\d+) boxes", clause, re.I):
            value += int(m[2]) * (1 if m[1].lower() == "receives" else -1)
        else:
            return None
    return str(value)


def solve_csv(task: Task, work: Path) -> str | None:
    match = re.search(
        r"How many units of product ['\"]([^'\"]+)['\"] were sold in months "
        r"(\d+) through (\d+) inclusive\?",
        task.instruction,
        re.I,
    )
    if not match:
        return None
    product, lo, hi = match[1], int(match[2]), int(match[3])
    if not 1 <= lo <= hi <= 12:
        return None
    path = work / "sales.csv"
    if not path.is_file():
        raise InsufficientData("Required sales.csv is missing")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        if not {"product", "month", "units"}.issubset(rows.fieldnames or []):
            raise InsufficientData("Required CSV columns are missing")
        total = 0
        for row in rows:
            if not row.get("product"):
                raise InsufficientData("CSV contains a missing product")
            if row["product"] != product:
                continue
            try:
                month = int(row["month"])
            except (ValueError, TypeError):
                raise InsufficientData("CSV contains missing or noninteger month") from None
            if not 1 <= month <= 12:
                raise InsufficientData("Invalid month in CSV")
            if lo <= month <= hi:
                try:
                    total += int(row["units"])
                except (ValueError, TypeError):
                    raise InsufficientData(
                        "Requested rows contain missing/noninteger units"
                    ) from None
    return str(total)


def solve_search(instruction: str, work: Path) -> str | None:
    direct = re.search(r"In which year was (.+?) born\?", instruction, re.I)
    employer = re.search(
        r"In which city is the headquarters of (.+?)[’']s employer\?", instruction, re.I
    )
    if not (direct or employer):
        return None
    records: dict[str, dict[str, set[str]]] = {}
    paths = sorted((work / "kb").rglob("*.md"))
    if not paths:
        raise InsufficientData("Required knowledge base is missing")
    for path in paths:
        lines = path.read_text().splitlines()
        heading = next((line[2:].strip() for line in lines if line.startswith("# ")), None)
        if not heading:
            return None
        fields = records.setdefault(heading.casefold(), {})
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key, value = key.strip().casefold(), value.strip()
                fields.setdefault(key, set()).add(value)

    def field(name, key):
        values = records.get(name, {}).get(key, set())
        if len(values) != 1:
            raise InsufficientData(f"Required field {key} is missing or conflicting")
        return next(iter(values))

    name = (direct or employer)[1].strip().casefold()
    if direct:
        value = field(name, "born")
        if re.fullmatch(r"\d{4}", value):
            return value
        raise InsufficientData("Birth year is absent or ambiguous")
    company = field(name, "employer").casefold()
    city = field(company, "headquarters")
    if not city:
        raise InsufficientData("Employer or headquarters is missing")
    return city
