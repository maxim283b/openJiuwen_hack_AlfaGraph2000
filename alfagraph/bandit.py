"""Cost-aware disjoint LinUCB, fit offline from randomized bandit feedback."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from alfagraph.settings import MODELS

FEATURE_NAMES = [
    "bias",
    "math",
    "search",
    "data_analysis",
    "coding",
    "instruction_chars",
    "files",
    "bytes",
    "rows",
    "records",
    "functions",
    "branches",
    "tests",
    "operations",
    "source_lines",
]
SCALES = {
    "instruction_chars": 100000,
    "files": 300,
    "bytes": 8000000,
    "rows": 10000,
    "records": 1000,
    "functions": 100,
    "branches": 100,
    "tests": 100,
    "operations": 32,
    "source_lines": 2000,
}


def vector(features):
    values = [1.0] + [float(features["kind"] == kind) for kind in FEATURE_NAMES[1:5]]
    values.extend(
        min(1.0, math.log1p(max(0, features.get(name, 0))) / math.log1p(scale))
        for name, scale in SCALES.items()
    )
    return values


def dot(left, right):
    return sum(a * b for a, b in zip(left, right, strict=True))


def matvec(matrix, value):
    return [dot(row, value) for row in matrix]


def resolve_config(value):
    path = Path(value)
    return path if path.is_absolute() else Path(__file__).resolve().parents[1] / path


@dataclass(frozen=True)
class Arm:
    name: str
    model: str
    tokens: int

    def __post_init__(self):
        if self.model not in MODELS or not isinstance(self.tokens, int) or self.tokens < 256:
            raise ValueError("Invalid bandit model or task token cap")

    def as_dict(self):
        return {"name": self.name, "model": self.model, "tokens": self.tokens}


def default_arms():
    return [
        Arm(f"m{i}-t{cap}", model, cap)
        for i, model in enumerate(
            [
                "qwen3.6-35b-a3b-noreason",
                "qwen3.8-27b-noreason",
                "qwen3.8-27b",
            ]
        )
        for cap in [1800, 3600]
    ]


@dataclass(frozen=True)
class Prediction:
    arm: Arm
    probability: float
    cost: float
    uncertainty: float

    @property
    def efficiency(self):
        return self.probability / self.cost

    def as_dict(self):
        return {
            **self.arm.as_dict(),
            "probability": round(self.probability, 6),
            "cost": round(self.cost, 3),
            "uncertainty": round(self.uncertainty, 6),
        }


class LinUCB:
    def __init__(self, arms, ridge=2.0, alpha=0.05, cost_alpha=0.1):
        if ridge <= 0 or not math.isfinite(ridge) or alpha < 0 or cost_alpha < 0:
            raise ValueError("Invalid bandit regularization or confidence parameters")
        self.arms = {arm.name: arm for arm in arms}
        if len(self.arms) != len(arms) or not self.arms:
            raise ValueError("Bandit requires unique actions")
        self.ridge, self.alpha, self.cost_alpha = ridge, alpha, cost_alpha
        size = len(FEATURE_NAMES)
        self.states = {}
        for arm in arms:
            self.states[arm.name] = {
                "count": 0,
                "inverse": [[float(i == j) / ridge for j in range(size)] for i in range(size)],
                "reward": [ridge * 0.5] + [0.0] * (size - 1),
                "cost": [ridge * arm.tokens * 0.3 / 1000] + [0.0] * (size - 1),
            }
        self.training = {}

    def update(self, arm_name, features, correct, cost):
        if correct not in {0, 1, False, True} or not math.isfinite(cost) or cost <= 0:
            raise ValueError("Bandit requires observed correctness and positive measured cost")
        x = vector(features)
        state = self.states[arm_name]
        z = matvec(state["inverse"], x)
        denominator = 1 + dot(x, z)
        # Sherman-Morrison update of (lambda I + sum x*x.T)^-1.
        state["inverse"] = [
            [value - z[i] * z[j] / denominator for j, value in enumerate(row)]
            for i, row in enumerate(state["inverse"])
        ]
        state["reward"] = [b + a * float(correct) for b, a in zip(state["reward"], x, strict=True)]
        state["cost"] = [b + a * cost / 1000 for b, a in zip(state["cost"], x, strict=True)]
        state["count"] += 1

    def predictions(self, features, ceiling=6000):
        x = vector(features)
        predictions = []
        for name, arm in self.arms.items():
            state = self.states[name]
            if arm.tokens > ceiling or state["count"] < 5:
                continue
            uncertainty = math.sqrt(max(0, dot(x, matvec(state["inverse"], x))))
            reward = dot(x, matvec(state["inverse"], state["reward"]))
            cost = dot(x, matvec(state["inverse"], state["cost"]))
            predictions.append(
                Prediction(
                    arm,
                    max(0.01, min(0.99, reward + self.alpha * uncertainty)),
                    max(128, (cost + self.cost_alpha * uncertainty) * 1000),
                    uncertainty,
                )
            )
        return predictions

    def choose(self, features, ceiling=6000):
        predictions = self.predictions(features, ceiling)
        return (
            max(predictions, key=lambda prediction: prediction.efficiency) if predictions else None
        )

    def as_dict(self):
        return {
            "schema": 1,
            "algorithm": "cost-aware-disjoint-linucb",
            "unit": "normalized_output_tokens",
            "features": FEATURE_NAMES,
            "ridge": self.ridge,
            "alpha": self.alpha,
            "cost_alpha": self.cost_alpha,
            "arms": [arm.as_dict() for arm in self.arms.values()],
            "states": self.states,
            "training": self.training,
        }

    @classmethod
    def load(cls, path):
        if not path:
            return None
        data = json.loads(resolve_config(path).read_text())
        if (
            data.get("schema") != 1
            or data.get("algorithm") != "cost-aware-disjoint-linucb"
            or data.get("features") != FEATURE_NAMES
            or data.get("unit") != "normalized_output_tokens"
        ):
            raise ValueError("Unsupported bandit artifact")
        policy = cls(
            [Arm(**arm) for arm in data["arms"]], data["ridge"], data["alpha"], data["cost_alpha"]
        )
        size = len(FEATURE_NAMES)
        for name in policy.arms:
            state = data["states"][name]
            values = [
                *state["reward"],
                *state["cost"],
                *(v for row in state["inverse"] for v in row),
            ]
            if (
                len(state["reward"]) != size
                or len(state["cost"]) != size
                or len(state["inverse"]) != size
                or any(len(row) != size for row in state["inverse"])
                or not all(math.isfinite(value) for value in values)
                or not isinstance(state["count"], int)
                or state["count"] < 0
            ):
                raise ValueError("Invalid bandit sufficient statistics")
        policy.states = data["states"]
        policy.training = data.get("training", {})
        return policy
