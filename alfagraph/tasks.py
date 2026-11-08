from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

TYPES = {"math", "search", "data_analysis", "coding"}


@dataclass
class Task:
    task_id: str
    kind: str
    reward: float
    instruction: str
    directory: Path
    workspace: Path
    check: str = ""


@dataclass
class Attempt:
    task: Task
    work: Path
    answer: str | None = None
    status: str = "pending"
    reason: str = ""
    local_tried: bool = False
    tests: dict[str, str] = field(default_factory=dict)
    observations: list[str] = field(default_factory=list)
    method: str = ""
    evidence_files: set[str] = field(default_factory=set)
    test_result: dict = field(default_factory=dict)
    review_pending: bool = False
    review_done: bool = False
    verification: dict = field(default_factory=dict)


def load_tasks(root: Path) -> list[Task]:
    """Only read public fields. Never pass task.json or expected to a solver."""
    root = root.resolve(strict=True)
    paths = (
        [root / "task.json"] if (root / "task.json").is_file() else sorted(root.glob("*/task.json"))
    )
    tasks = []
    for path in paths:
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Task path escapes tasks directory: {path.name}")
        obj = json.loads(path.read_text())
        kind = obj["task_type"]
        reward = float(obj["reward"])
        instruction = obj["instruction"]
        if kind not in TYPES or not math.isfinite(reward) or reward < 0:
            raise ValueError(f"Invalid task type/reward: {path.parent.name}")
        if not isinstance(instruction, str) or len(instruction) > 100_000:
            raise ValueError(f"Invalid instruction: {path.parent.name}")
        workspace = path.parent / "workspace"
        tasks.append(
            Task(
                path.parent.name,
                kind,
                reward,
                instruction,
                path.parent,
                workspace if workspace.is_dir() else path.parent,
                str(obj.get("check", "")),
            )
        )
    if not tasks:
        raise ValueError("No task.json files found")
    return tasks


def normalize_answer(task: Task, value: object) -> str:
    answer = str(value).strip()
    if not answer or "\n" in answer or "\r" in answer or len(answer) > 500:
        raise ValueError("Answer must be a single nonempty line")
    if task.check == "exact_int" or task.kind in {"math", "data_analysis"}:
        if not re.fullmatch(r"[+-]?\d+", answer):
            raise ValueError("Expected an integer answer")
        answer = str(int(answer))
    if task.kind == "coding" and answer != "DONE":
        raise ValueError("Coding answer must be DONE")
    return answer


def atomic_write(path: Path, text: str) -> None:
    """Commit each answer independently; interruption cannot leave half a result."""
    fd, temporary = tempfile.mkstemp(prefix=".answer-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
