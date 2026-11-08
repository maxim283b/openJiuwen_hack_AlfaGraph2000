from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass

from alfagraph.tasks import Task


@dataclass
class Estimate:
    tokens: float
    success_probability: float
    features: dict
    expected_spend: float | None = None
    action: object | None = None

    @property
    def cost(self):
        return self.tokens if self.expected_spend is None else self.expected_spend

    def priority(self, reward: float) -> float:
        return reward * self.success_probability / max(1, self.cost)


class Planner:
    """Transparent starting estimator, calibrated online from measured trajectories."""

    BASE = {
        "math": (450, 0.9),
        "search": (800, 0.85),
        "data_analysis": (1200, 0.85),
        "coding": (2600, 0.7),
    }

    def __init__(
        self, profile=None, bandit=None, ceiling=6000, selection=None, total_budget=100000
    ):
        self.history: dict[str, list[tuple[float, bool]]] = {}
        self.feature_cache = {}
        self.profile = profile
        self.bandit, self.ceiling = bandit, ceiling
        self.selection, self.total_budget = selection, total_budget
        self.feedback = {}
        self.initial_pending = None

    def features(self, task):
        key = (str(task.workspace), task.kind, task.instruction)
        if key in self.feature_cache:
            return self.feature_cache[key]
        files = [
            p
            for p in task.workspace.rglob("*")
            if p.is_file()
            and p.name not in {"task.json", "answer.txt"}
            and not any(part in {"__pycache__", ".harness", ".git"} for part in p.parts)
        ]
        features = {
            "kind": task.kind,
            "instruction_chars": len(task.instruction),
            "files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
            "operations": len(
                re.findall(
                    r"\b(?:ships|receives|split|doubled|sum|product|median|filter|group)\b|[+*/]",
                    task.instruction,
                    re.I,
                )
            ),
            "rows": 0,
            "records": 0,
            "functions": 0,
            "branches": 0,
            "source_lines": 0,
            "tests": 0,
        }
        for path in files:
            if path.suffix not in {".py", ".md", ".csv", ".json"}:
                continue
            try:
                with path.open(errors="replace") as stream:
                    text = stream.read(1_000_000)
                if path.suffix == ".csv":
                    features["rows"] += max(0, len(text.splitlines()) - 1)
                elif path.suffix == ".md":
                    features["records"] += len(re.findall(r"^# ", text, re.M))
                elif path.suffix == ".py":
                    nodes = list(ast.walk(ast.parse(text[:100_000])))
                    functions = sum(
                        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in nodes
                    )
                    if path.name.startswith("test") or "tests" in path.parts:
                        features["tests"] += functions
                    else:
                        features["functions"] += functions
                        features["source_lines"] += len(text.splitlines())
                        features["branches"] += sum(
                            isinstance(n, (ast.If, ast.For, ast.While, ast.Try, ast.Match))
                            for n in nodes
                        )
                elif path.suffix == ".json":
                    obj = json.loads(text)
                    if isinstance(obj, dict) and isinstance(obj.get("operations"), list):
                        features["operations"] += len(obj["operations"])
            except (OSError, SyntaxError, RecursionError, ValueError):
                continue
        self.feature_cache[key] = features
        return features

    def estimate(self, task: Task) -> Estimate:
        features = self.features(task)
        baseline, probability = self.BASE[task.kind]
        if task.kind == "math":
            complexity = 0.8 + 0.12 * min(12, features["operations"])
        elif task.kind == "search":
            hops = max(1, len(re.findall(r"employer|mentor|headquarters", task.instruction, re.I)))
            if depth := re.search(r"Mentor exactly (\d+) times", task.instruction, re.I):
                hops += max(0, min(10, int(depth[1])) - 1)
            complexity = 0.6 + 0.25 * hops + 0.07 * math.log1p(features["records"])
        elif task.kind == "data_analysis":
            complexity = (
                0.7 + 0.16 * math.log1p(features["rows"] / 100) + 0.12 * features["operations"]
            )
        else:
            complexity = (
                0.65
                + 0.04 * features["functions"]
                + 0.06 * features["branches"]
                + 0.08 * math.log1p(features["source_lines"])
                + 0.05 * features["tests"]
            )
        complexity += 0.1 * math.log1p(features["instruction_chars"] / 500)
        complexity = min(4, complexity)
        predicted = baseline * complexity
        observations = self.history.get(task.kind, [])
        if observations:
            successes = [cost for cost, ok in observations if ok]
            if successes:
                predicted = 0.5 * predicted + 0.5 * sum(successes) / len(successes)
            # An interrupted/failed run is a lower bound, not a cheap solved example.
            predicted = max(
                predicted, max((cost for cost, ok in observations if not ok), default=0)
            )
            probability = (sum(ok for _, ok in observations) + 2 * probability) / (
                len(observations) + 2
            )
        spend = None
        if self.profile:
            spend, probability = self.profile.predict(task.kind, features, predicted, probability)
        action = None
        if self.bandit and (route := self.bandit.choose(features, self.ceiling)):
            spend, probability, action = route.cost, route.probability, route.arm
        # Measured spend guides ordering; it does not shrink the reservation needed
        # to fit schemas, context and a complete tool call.
        return Estimate(predicted, probability, features, spend, action)

    def record(self, task: Task, cost: float, solved: bool):
        if cost > 0:
            predicted = self.estimate(task).cost
            state = self.feedback.setdefault(
                task.kind, {"cost": 0, "predicted": 0, "attempts": 0, "submitted": 0}
            )
            state["cost"] += cost
            state["predicted"] += predicted
            state["attempts"] += 1
            state["submitted"] += bool(solved)
            self.history.setdefault(task.kind, []).append((max(32, cost), solved))

    def choose(self, tasks: list[Task], remaining: float) -> Task | None:
        eligible = [t for t in tasks if t.reward > 0 and remaining >= 64]
        if not eligible:
            return None
        if self.selection:
            self.initial_pending = self.initial_pending or len(eligible)
            estimates = {t.task_id: self.estimate(t) for t in eligible}
            queue_cost = sum(e.cost for e in estimates.values())

            def score(task):
                e = estimates[task.task_id]
                return self.selection.score(
                    task.reward,
                    e.success_probability,
                    e.cost,
                    e.features,
                    remaining=remaining,
                    total=self.total_budget,
                    queue_cost=queue_cost,
                    queue_fraction=len(eligible) / self.initial_pending,
                    feedback=self.feedback,
                )

            return max(eligible, key=score)
        return max(eligible, key=lambda t: self.estimate(t).priority(t.reward))
