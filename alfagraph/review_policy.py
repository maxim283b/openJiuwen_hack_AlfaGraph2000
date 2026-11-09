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
        from alfagraph.review_features import EXTRA

        expected = FEATURES if data["schema"] == 1 else FEATURES + EXTRA
        if data["features"] != expected or data["schema"] not in {1, 2}:
            raise ValueError("Incompatible review policy")
        self.data = data

    @classmethod
    def load(cls, path):
        return cls(json.loads(resolve_config(path).read_text())) if path else None

    def predict(self, attempt, spent, state=None):
        if self.data["schema"] == 2:
            from alfagraph.review_features import raw_text, rich_features, tokens

            x = rich_features(attempt, spent, state)
            counts = __import__("collections").Counter(tokens(raw_text(attempt)))
            text = {
                w: (1 + math.log(n)) * self.data.get("idf", {}).get(w, 0)
                for w, n in counts.items()
                if w in self.data.get("text_weights", {})
            }
            norm = math.sqrt(sum(v * v for v in text.values())) or 1
        else:
            x = features(attempt, spent, state)
            text, norm = {}, 1

        def head(name):
            value = sum(a * b for a, b in zip(x, self.data["weights"][name], strict=True))
            index = {"fix": 0, "harm": 1, "log_cost": 2}[name]
            return value + sum(
                v / norm * self.data["text_weights"][w][index] for w, v in text.items()
            )

        fix = min(1, max(0, head("fix")))
        harm = min(1, max(0.01, head("harm")))
        cost = min(3200, max(150, math.expm1(min(9, head("log_cost")))))
        supported = self.data["support"].get(attempt.task.kind, 0) >= 3
        review_kinds = self.data.get("review_kinds")
        if review_kinds is not None:
            supported = supported and attempt.task.kind in review_kinds
        gain = attempt.task.reward * (fix - harm) if supported else 0
        return {
            "fix_probability": fix,
            "harm_probability": harm,
            "expected_gain": gain,
            "cost": cost,
            "priority": gain / cost,
            "supported": supported,
        }

    def choose(self, attempts, remaining, task_spent, states, task_ceiling=None, predictions=None):
        ranked = []
        for a in attempts:
            prediction = (
                predictions[a.task.task_id]
                if predictions is not None
                else self.predict(
                    a, task_spent.get(a.task.task_id, 0) / 10, states.get(a.task.task_id, {})
                )
            )
            available = remaining
            if task_ceiling is not None:
                available = min(available, task_ceiling - task_spent.get(a.task.task_id, 0) / 10)
            if prediction["expected_gain"] > 0 and prediction["cost"] <= available:
                ranked.append((prediction["priority"], a.task.task_id, a.task, prediction))
        if not ranked:
            return None
        _, _, task, prediction = max(ranked, key=lambda r: (r[0], r[1]))
        return task, prediction
