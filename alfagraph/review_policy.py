"""Expected score gain and review cost from public candidate features."""

from __future__ import annotations

import json
import math
import re

from alfagraph.bandit import resolve_config
from alfagraph.context import workspace_files

FEATURES = [
    "bias",
    "math",
    "search",
    "data_analysis",
    "coding",
    "has_answer",
    "answer_words",
    "instruction_length",
    "workspace_bytes",
    "producer_cost",
    "steps",
    "errors",
    "truncated",
    "negation",
    "numbers",
]


def features(attempt, spent, state=None):
    state = state or {}
    text = attempt.task.instruction.lower()
    size = sum(p.stat().st_size for p in workspace_files(attempt.work))

    def scale(v, limit):
        return min(2, math.log1p(max(0, v)) / math.log1p(limit))

    return [
        1.0,
        *[float(attempt.task.kind == k) for k in FEATURES[1:5]],
        float(attempt.answer is not None),
        scale(len((attempt.answer or "").split()), 20),
        scale(len(text), 2000),
        scale(size, 20000),
        scale(spent, 3000),
        scale(state.get("steps", 0), 8),
        scale(state.get("errors", 0), 4),
        scale(state.get("truncated", 0), 4),
        float(bool(re.search(r"\b(not|never|except|without)\b|не |кроме", text))),
        scale(len(re.findall(r"\d+", text)), 20),
    ]


class ReviewPolicy:
    def __init__(self, data):
        if data["features"] != FEATURES or data["schema"] != 1:
            raise ValueError("Incompatible review policy")
        self.data = data

    @classmethod
    def load(cls, path):
        return cls(json.loads(resolve_config(path).read_text())) if path else None

    def predict(self, attempt, spent, state=None):
        x = features(attempt, spent, state)

        def head(name):
            return sum(a * b for a, b in zip(x, self.data["weights"][name], strict=True))

        fix = min(1, max(0, head("fix")))
        harm = min(1, max(0.01, head("harm")))
        cost = min(3200, max(150, math.expm1(min(9, head("log_cost")))))
        supported = self.data["support"].get(attempt.task.kind, 0) >= 3
        gain = attempt.task.reward * (fix - harm) if supported else 0
        return {
            "fix_probability": fix,
            "harm_probability": harm,
            "expected_gain": gain,
            "cost": cost,
            "priority": gain / cost,
            "supported": supported,
        }

    def choose(self, attempts, remaining, task_spent, states):
        ranked = []
        for a in attempts:
            prediction = self.predict(
                a, task_spent.get(a.task.task_id, 0) / 10, states.get(a.task.task_id, {})
            )
            if prediction["expected_gain"] > 0 and prediction["cost"] <= remaining:
                ranked.append((prediction["priority"], a.task.task_id, a.task, prediction))
        if not ranked:
            return None
        _, _, task, prediction = max(ranked, key=lambda r: (r[0], r[1]))
        return task, prediction
