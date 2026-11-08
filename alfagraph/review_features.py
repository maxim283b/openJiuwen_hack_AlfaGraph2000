"""Task-derived structure and learned text inputs; never reads labels or task IDs."""

from __future__ import annotations

import ast
import csv
import io
import math
import re
from collections import Counter

from alfagraph.context import workspace_files
from alfagraph.review_policy import features as basic_features

EXTRA = [
    "files",
    "python_files",
    "csv_files",
    "text_files",
    "source_lines",
    "functions",
    "branches",
    "loops",
    "calls",
    "imports",
    "exceptions",
    "nesting",
    "assertions",
    "test_lines",
    "csv_rows",
    "csv_columns",
    "missing_cells",
    "numeric_cells",
    "sentences",
    "documents",
    "unique_words",
    "lexical_overlap",
    "answer_in_data",
    "answer_numeric",
    "answer_digits",
    "answer_length",
    "comparisons",
    "aggregations",
    "unit_words",
    "date_mentions",
    "candidate_present",
    "execution_errors",
    "reads",
    "executions",
    "edits",
    "test_runs",
    "empty_calls",
    "provided_tests_ok",
    "provided_failures",
    "producer_reasoning",
    "producer_cap",
    "instruction_workspace_ratio",
]


def raw_text(attempt):
    parts = [attempt.task.instruction, attempt.answer or ""]
    for p in workspace_files(attempt.work)[:20]:
        if p.suffix in {".py", ".md", ".txt", ".csv", ".json"}:
            text = p.read_text(errors="replace")[:2000]
            # Bound text representation; identifiers/answers stay ordinary input tokens.
            parts.append(text)
    return "\n".join(parts)[:18000]


def tokens(text):
    words = re.findall(r"[a-zа-я_]+|\d+", text.lower())
    words = ["<num>" if w.isdigit() else w for w in words]
    return words + [a + " " + b for a, b in zip(words, words[1:])]


def rich_features(attempt, spent, state=None):
    state = state or {}
    files = workspace_files(attempt.work)
    d = Counter()
    content = []
    for p in files:
        d["files"] += 1
        if p.suffix not in {".py", ".md", ".txt", ".csv", ".json"}:
            continue
        text = p.read_text(errors="replace")[:60000]
        content.append(text)
        if p.suffix == ".py":
            d["python_files"] += 1
            d["source_lines"] += len(text.splitlines())
            if p.name.startswith("test"):
                d["test_lines"] += len(text.splitlines())
            try:
                tree = ast.parse(text)

                def depth(node):
                    children = list(ast.iter_child_nodes(node))
                    return 1 + max((depth(x) for x in children), default=0)

                d["nesting"] = max(d["nesting"], depth(tree))
                for n in ast.walk(tree):
                    for key, classes in [
                        ("functions", (ast.FunctionDef, ast.AsyncFunctionDef)),
                        ("branches", (ast.If, ast.IfExp)),
                        ("loops", (ast.For, ast.While, ast.comprehension)),
                        ("calls", (ast.Call,)),
                        ("imports", (ast.Import, ast.ImportFrom)),
                        ("exceptions", (ast.Try, ast.Raise)),
                        ("assertions", (ast.Assert,)),
                    ]:
                        d[key] += isinstance(n, classes)
            except (SyntaxError, RecursionError):
                d["execution_errors"] += 1
        elif p.suffix == ".csv":
            d["csv_files"] += 1
            rows = list(csv.reader(io.StringIO(text)))
            d["csv_rows"] += max(0, len(rows) - 1)
            d["csv_columns"] = max(d["csv_columns"], len(rows[0]) if rows else 0)
            for row in rows[1:]:
                for cell in row:
                    d["missing_cells"] += cell.strip().lower() in {"", "na", "null", "nan", "none"}
                    d["numeric_cells"] += bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", cell.strip()))
        else:
            d["text_files"] += 1
        d["sentences"] += len(re.findall(r"[.!?](?:\s|$)", text))
        d["documents"] += len(re.findall(r"^#", text, re.M))
    data = "\n".join(content)
    question = attempt.task.instruction.lower()
    answer = attempt.answer or ""
    qwords = set(re.findall(r"\w+", question))
    words = set(re.findall(r"\w+", data.lower()))
    d["unique_words"] = len(words)
    d["lexical_overlap"] = len(qwords & words) / max(1, len(qwords))
    d["answer_in_data"] = float(bool(answer) and answer in data)
    d["answer_numeric"] = float(bool(re.fullmatch(r"[-+]?\d+", answer)))
    d["answer_digits"] = sum(c.isdigit() for c in answer)
    d["answer_length"] = len(answer)
    d["comparisons"] = len(
        re.findall(
            r"\b(more|less|greater|largest|smallest|before|after|than|difference)\b", question
        )
    )
    d["aggregations"] = len(
        re.findall(r"\b(total|sum|count|average|maximum|minimum|mean|how many)\b", question)
    )
    d["unit_words"] = len(
        re.findall(
            r"\b(percent|dollars|cents|million|billion|hours|minutes|meters|miles|years)\b",
            question,
        )
    )
    d["date_mentions"] = len(re.findall(r"\b(?:19|20)\d{2}\b", question))
    d["candidate_present"] = float(attempt.answer is not None)
    d["execution_errors"] += state.get("errors", 0)
    for key, source in [
        ("reads", "reads"),
        ("executions", "execs"),
        ("edits", "edits"),
        ("test_runs", "tests"),
    ]:
        d[key] = state.get(source, 0)
    d["empty_calls"] = state.get("empty", 0)
    d["provided_tests_ok"] = float(attempt.test_result.get("ok", False))
    d["provided_failures"] = attempt.test_result.get("failures", 0) + attempt.test_result.get(
        "errors", 0
    )
    d["producer_reasoning"] = float(
        bool(state.get("producer_model")) and not state["producer_model"].endswith("noreason")
    )
    d["producer_cap"] = state.get("task_limit", 0)
    d["instruction_workspace_ratio"] = len(question) / max(1, len(data))
    binary = {
        "lexical_overlap",
        "answer_in_data",
        "answer_numeric",
        "candidate_present",
        "provided_tests_ok",
        "producer_reasoning",
        "instruction_workspace_ratio",
    }
    return basic_features(attempt, spent, state) + [
        min(2, d[k]) if k in binary else min(3, math.log1p(d[k]) / math.log(101)) for k in EXTRA
    ]
