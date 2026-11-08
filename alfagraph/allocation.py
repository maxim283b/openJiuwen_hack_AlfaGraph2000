"""Per-task token tranches, unlocked by bounded, externally observed progress."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field


@dataclass
class Progress:
    files: set[str] = field(default_factory=set)
    candidate: bool = False
    edited: bool = False
    executed: bool = False
    best_failures: int | None = None
    points: int = 0
    last_signal: str = ""

    def tests(self, result: dict, baseline=False):
        if not result.get("tests") or not all(k in result for k in ("failures", "errors")):
            return
        failures = int(result["failures"]) + int(result["errors"])
        if not baseline and self.best_failures is not None and failures < self.best_failures:
            self.points += 3
            self.last_signal = "fewer_failing_tests"
        if self.best_failures is None or failures < self.best_failures:
            self.best_failures = failures

    def observe(self, name: str, args: dict, result, work):
        if isinstance(result, dict) and result.get("error"):
            return
        if name == "read_file" and isinstance(result, str) and result:
            path = (work / args["path"]).resolve()
            if str(path) not in self.files:
                self.files.add(str(path))
                if len(self.files) <= 3:
                    self.points += 1
                    self.last_signal = "new_file_read"
        elif name == "search_files" and isinstance(result, dict):
            for match in result.get("matches", []):
                self.observe("read_file", {"path": match["path"]}, match["text"], work)
        elif (
            name in {"run_python", "compute_answer"}
            and isinstance(result, dict)
            and result.get("returncode") == 0
        ):
            lines = result.get("stdout", "").strip().splitlines()
            if lines and not self.executed:
                self.executed = True
                self.points += 1
                self.last_signal = "successful_execution"
            if lines and re.fullmatch(r"[+-]?\d+", lines[-1]) and not self.candidate:
                self.candidate = True
                self.points += 2
                self.last_signal = "computed_candidate"
        elif name in {"edit_file", "write_file"} and (
            result == "saved" or isinstance(result, dict) and result.get("saved")
        ):
            # One exploration credit for the first edit; arbitrary further edits
            # cannot unlock more budget without a measurable test improvement.
            if not self.edited:
                self.edited = True
                self.points += 1
                self.last_signal = "first_source_edit"
            if isinstance(result, dict) and result.get("tests"):
                self.tests(result["tests"])
        elif name == "run_tests" and isinstance(result, dict):
            self.tests(result)


@dataclass
class Allocation:
    mode: str
    predicted: float
    limit: int
    hard_limit: int
    patience: int = 3
    stage: int = 1
    granted_points: int = 0
    last_points: int = 0
    stale_steps: int = 0
    stopped: str = ""
    format_retries: int = 0
    retry_pending: bool = False
    route_model: str | None = None
    route_action: str | None = None

    @classmethod
    def create(cls, estimate, ceiling: int, mode="adaptive", patience=3):
        if mode == "fixed":
            return cls(mode, estimate.tokens, ceiling, ceiling, patience)
        # Include room for prompt/schema framing on the final submission call.
        hard = min(ceiling, math.ceil(max(1000, estimate.tokens * 1.8)))
        initial = min(hard, math.ceil(max(450, estimate.tokens * 0.45)))
        return cls(mode, estimate.tokens, initial, hard, patience)

    def extend(self, needed: int, progress: Progress, spent: float) -> str | None:
        if self.mode == "fixed" or needed <= self.limit or needed > self.hard_limit:
            return None
        if spent == 0 and self.stage == 1:
            # Tool schemas and prompt framing must fit even before the first action.
            self.limit = needed
            return "initial_prompt_capacity"
        if self.retry_pending:
            self.limit = min(self.hard_limit, max(needed, self.limit + 200))
            self.retry_pending = False
            self.stage += 1
            return "bounded_format_retry"
        if progress.points <= self.granted_points:
            return None
        self.limit = min(
            self.hard_limit, max(needed, self.limit + math.ceil(max(200, self.predicted * 0.4)))
        )
        self.granted_points = progress.points
        self.stage += 1
        return progress.last_signal

    def after_step(self, progress: Progress):
        self.stale_steps = 0 if progress.points > self.last_points else self.stale_steps + 1
        self.last_points = progress.points
        if self.mode == "adaptive" and self.stale_steps >= self.patience:
            self.stopped = "No measurable progress"

    def output_limit(self, model: str, kind: str) -> int:
        if self.mode == "fixed":
            return 768 if model.endswith("noreason") else 1536
        if model.endswith("noreason"):
            base = {"math": 512, "search": 384, "data_analysis": 768, "coding": 1024}[kind]
        else:
            base = 1536 if kind == "coding" else 1024
        return base * (2 if self.format_retries else 1)

    def request_format_retry(self):
        if self.format_retries:
            return False
        self.format_retries = 1
        self.retry_pending = True
        return True

    def minimum_output(self, kind: str) -> int:
        if self.mode == "fixed":
            return 32
        return {"math": 256, "search": 192, "data_analysis": 512, "coding": 640}[kind]

    def report(self):
        return {
            "mode": self.mode,
            "predicted": round(self.predicted, 2),
            "limit": self.limit,
            "hard_limit": self.hard_limit,
            "stage": self.stage,
            "granted_points": self.granted_points,
            "stale_steps": self.stale_steps,
            "stopped": self.stopped,
            "format_retries": self.format_retries,
            "route_model": self.route_model,
            "route_action": self.route_action,
        }
