"""Frozen state-dependent task ranking; fitting stays outside the harness."""

from __future__ import annotations

import hashlib
import json
import math

from alfagraph.bandit import resolve_config

KINDS = ("math", "search", "data_analysis", "coding")
STATIC_NAMES = [
    "log_reward",
    "log_probability",
    "log_cost",
    *KINDS,
    "instruction_size",
    "bytes_size",
    "rows_size",
]
FEATURE_NAMES = [
    *STATIC_NAMES,
    "affordability",
    "cost_feedback",
    "submission_feedback",
    *["budget_" + name for name in STATIC_NAMES[:3]],
    *["pressure_" + name for name in STATIC_NAMES[:3]],
    *["budget_" + kind for kind in KINDS],
    *["pressure_" + kind for kind in KINDS],
    "queue_cost",
]


def static_features(reward, probability, cost, features):
    return [
        min(2, math.log1p(max(0, reward)) / math.log(6)),
        max(-1, math.log(max(0.01, min(0.99, probability))) / 5),
        min(2, math.log1p(max(128, cost) / 128) / math.log(33)),
        *[float(features["kind"] == kind) for kind in KINDS],
        min(1, math.log1p(features.get("instruction_chars", 0)) / math.log1p(100000)),
        min(1, math.log1p(features.get("bytes", 0)) / math.log1p(8000000)),
        min(1, math.log1p(features.get("rows", 0)) / math.log1p(10000)),
    ]


def feature_vector(static, cost, kind, remaining, total, queue_cost, queue_fraction, feedback):
    budget = min(1, max(0, remaining / max(1, total)))
    pressure = min(2, queue_cost / max(64, remaining)) / 2
    observed = feedback.get(kind, {})
    cost_ratio = max(
        -1,
        min(1, math.log((1000 + observed.get("cost", 0)) / (1000 + observed.get("predicted", 0)))),
    )
    submission = (observed.get("submitted", 0) + 1) / (observed.get("attempts", 0) + 2) - 0.5
    return [
        *static,
        min(2, cost / max(64, remaining)),
        cost_ratio,
        submission,
        *[budget * v for v in static[:3]],
        *[pressure * v for v in static[:3]],
        *[budget * v for v in static[3:7]],
        *[pressure * v for v in static[3:7]],
        queue_fraction * static[2],
    ]


class SelectionPolicy:
    def __init__(self, weights, training=None):
        if len(weights) != len(FEATURE_NAMES) or not all(math.isfinite(w) for w in weights):
            raise ValueError("Invalid selection policy weights")
        self.weights = tuple(float(w) for w in weights)
        self.training = training or {}

    def score(
        self,
        reward,
        probability,
        cost,
        features,
        *,
        remaining,
        total,
        queue_cost,
        queue_fraction,
        feedback,
    ):
        base = (
            math.log(max(0.0001, reward))
            + math.log(max(0.01, probability))
            - math.log(max(128, cost))
        )
        static = static_features(reward, probability, cost, features)
        values = feature_vector(
            static, cost, features["kind"], remaining, total, queue_cost, queue_fraction, feedback
        )
        return base + sum(w * x for w, x in zip(self.weights, values, strict=True))

    def as_dict(self):
        return {
            "schema": 1,
            "algorithm": "cem-state-ranking",
            "feature_names": FEATURE_NAMES,
            "weights": list(self.weights),
            "training": self.training,
        }

    @classmethod
    def load(cls, path, router=None):
        if not path:
            return None
        data = json.loads(resolve_config(path).read_text())
        if (
            data.get("schema") != 1
            or data.get("algorithm") != "cem-state-ranking"
            or data.get("feature_names") != FEATURE_NAMES
        ):
            raise ValueError("Unsupported selection policy")
        expected = data.get("training", {}).get("router_sha256")
        if expected and (
            not router
            or hashlib.sha256(resolve_config(router).read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Selection policy requires its frozen model router")
        return cls(data["weights"], data.get("training", {}))
