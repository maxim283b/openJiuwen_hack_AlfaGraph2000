"""Empirical future transitions and budgeted planning; no local neural model."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

from alfagraph.bandit import resolve_config

KINDS = ("math", "search", "data_analysis", "coding")
COUNTERS = ("steps", "spent", "errors", "truncated", "reads", "execs", "edits", "tests", "empty")
FEATURE_NAMES = [
    *KINDS,
    "instruction",
    "bytes",
    "rows",
    "operations",
    "functions",
    "branches",
    "source_lines",
    "visible_tests",
    *COUNTERS,
]


@dataclass
class StepState:
    steps: int = 0
    spent: float = 0
    errors: int = 0
    truncated: int = 0
    reads: int = 0
    execs: int = 0
    edits: int = 0
    tests: int = 0
    empty: int = 0

    def call(self, cost, finish_reason, has_calls):
        self.steps += 1
        self.spent += cost
        self.truncated += finish_reason == "length"
        self.empty += not has_calls and finish_reason != "length"

    def tool(self, name, diagnostic):
        self.errors += bool(diagnostic.get("error") or diagnostic.get("returncode"))
        self.reads += name in {"read_file", "search_files", "list_files"}
        self.execs += name in {"compute_answer", "run_python"}
        self.edits += name in {"write_file", "edit_file"}
        self.tests += name == "run_tests"

    def as_dict(self):
        return asdict(self)

    def successor(self, before, after, cost):
        values = {name: getattr(self, name) + after[name] - before[name] for name in COUNTERS}
        values["spent"] = self.spent + cost
        return StepState(**values)


def vector(features, state):
    def size(value, ceiling):
        return min(2, math.log1p(max(0, value)) / math.log1p(ceiling))

    return [
        *[float(features["kind"] == kind) for kind in KINDS],
        *[
            size(features.get(name, 0), ceiling)
            for name, ceiling in [
                ("instruction_chars", 2000),
                ("bytes", 100000),
                ("rows", 1000),
                ("operations", 12),
                ("functions", 8),
                ("branches", 12),
                ("source_lines", 200),
                ("tests", 8),
            ]
        ],
        min(2, state.steps / 4),
        min(2, state.spent / 1800),
        *[min(2, getattr(state, name) / 3) for name in COUNTERS[2:]],
    ]


@dataclass
class Forecast:
    probability: float
    cost: float
    p80: float
    next_cost: float
    steps: float
    neighbors: list


class FutureModel:
    def __init__(self, rows, training=None, neighbors=24):
        self.rows, self.training, self.neighbors = rows, training or {}, neighbors
        self.cache = {}

    @classmethod
    def load(cls, path, router):
        if not path:
            return None
        data = json.loads(resolve_config(path).read_text())
        if data.get("schema") != 1 or data.get("features") != FEATURE_NAMES:
            raise ValueError("Unsupported future-state model")
        if (
            not router
            or hashlib.sha256(resolve_config(router).read_bytes()).hexdigest()
            != data["training"]["router_sha256"]
        ):
            raise ValueError("Future-state model requires its frozen router")
        rows = data["rows"]
        for row in rows:
            if len(row["x"]) != len(FEATURE_NAMES) or not all(math.isfinite(v) for v in row["x"]):
                raise ValueError("Invalid future-state features")
            if not row["costs"] or any(not math.isfinite(c) or c <= 0 for c in row["costs"]):
                raise ValueError("Invalid observed future costs")
            if len(row["states"]) != len(row["costs"]) or row["correct"] not in [0, 1]:
                raise ValueError("Invalid observed future trajectory")
        return cls(rows, data["training"], data["neighbors"])

    def predict(self, features, state, arm, prior=0.7):
        x = vector(features, state)
        key = (tuple(x), arm.model, arm.tokens, round(prior, 4))
        if key in self.cache:
            return self.cache[key]
        candidates = []
        for row in self.rows:
            # Transition evidence must come from this model/cap and task type.
            if (
                row["model"] != arm.model
                or row["cap"] != arm.tokens
                or row["kind"] != features["kind"]
            ):
                continue
            # An unstarted state and a surviving prefix have different hazards.
            if bool(row["state"]["steps"]) != bool(state.steps):
                continue
            distance = sum((a - b) ** 2 for a, b in zip(x, row["x"], strict=True))
            candidates.append((distance, row))
        if not candidates:
            return None
        closest, used = [], set()
        for item in sorted(candidates, key=lambda item: item[0]):
            series = item[1].get("series", id(item[1]))
            if series in used:
                continue
            used.add(series)
            closest.append(item)
            if len(closest) == self.neighbors:
                break
        raw = [1 / (0.15 + distance) for distance, _ in closest]
        weights = [w / sum(raw) for w in raw]
        neighbors = [(w, row) for w, (_, row) in zip(weights, closest, strict=True)]
        cost = sum(w * sum(row["costs"]) for w, row in neighbors)
        effective = 1 / sum(w * w for w in weights)
        available = max(0, arm.tokens - state.spent)
        empirical = sum(
            w * row["correct"] * (sum(row["costs"]) <= available) for w, row in neighbors
        )
        probability = (effective * empirical + 4 * prior) / (effective + 4)
        cumulative, p80 = 0, cost
        for w, row in sorted(neighbors, key=lambda item: sum(item[1]["costs"])):
            cumulative += w
            if cumulative >= 0.8:
                p80 = sum(row["costs"])
                break
        result = Forecast(
            probability,
            max(32, cost),
            max(32, p80),
            sum(w * row["costs"][0] for w, row in neighbors),
            sum(w * len(row["costs"]) for w, row in neighbors),
            neighbors,
        )
        self.cache[key] = result
        return result


def portfolio(items, budget):
    """Fractional-knapsack terminal value, used only as an approximate critic."""
    value = 0.0
    for reward, cost in sorted(items, key=lambda item: item[0] / max(32, item[1]), reverse=True):
        fraction = min(1, max(0, budget) / max(32, cost))
        value += reward * fraction
        budget -= cost * fraction
        if budget <= 0:
            break
    return value


class ForecastPolicy:
    MODES = {"step", "density", "knapsack", "mpc"}

    def __init__(self, model, mode):
        if mode not in self.MODES:
            raise ValueError("Unsupported forecast policy")
        self.model, self.mode = model, mode
        self.last = {}

    def choose(self, tasks, remaining, planner, sessions):
        if not tasks or remaining < 64:
            return None
        if self.mode == "step":
            chosen = planner.choose(tasks, remaining)
            self.last = {"policy": "step", "predicted_next_tokens": 0}
            return chosen
        estimates, forecasts, states = {}, {}, {}
        for task in tasks:
            estimate = planner.estimate(task)
            state = sessions[task.task_id].state if task.task_id in sessions else StepState()
            forecast = (
                self.model.predict(
                    estimate.features, state, estimate.action, estimate.success_probability
                )
                if estimate.action
                else None
            )
            estimates[task.task_id], states[task.task_id] = estimate, state
            forecasts[task.task_id] = forecast or Forecast(
                estimate.success_probability,
                max(32, estimate.cost),
                max(32, estimate.cost),
                max(32, estimate.cost),
                1,
                [],
            )

        def value(task):
            f = forecasts[task.task_id]
            return task.reward * f.probability

        def density(task):
            return value(task) / forecasts[task.task_id].cost

        chosen = max(tasks, key=density)
        if self.mode == "knapsack":
            # Risk-adjusted 0/1 portfolio: p80 remaining cost, at most 512 bins.
            unit = max(32, remaining / 512)
            capacity = int(remaining // unit)
            dp = [(0.0, ()) for _ in range(capacity + 1)]
            for i, task in enumerate(tasks):
                cost = max(1, math.ceil(forecasts[task.task_id].p80 / unit))
                for b in range(capacity, cost - 1, -1):
                    previous, indices = dp[b - cost]
                    if previous + value(task) > dp[b][0]:
                        dp[b] = (previous + value(task), (*indices, i))
            selected = dp[-1][1]
            if selected:
                chosen = max((tasks[i] for i in selected), key=density)
        elif self.mode == "mpc":
            # One empirical next-step transition, then a portfolio value function.
            candidates = sorted(tasks, key=density, reverse=True)[:8]
            best = -math.inf
            for task in candidates:
                f, e, state = forecasts[task.task_id], estimates[task.task_id], states[task.task_id]
                others = [(value(t), forecasts[t.task_id].cost) for t in tasks if t is not task]
                outcomes = []
                for weight, row in f.neighbors:
                    cost = row["costs"][0]
                    if cost > remaining:
                        outcomes.append((weight, 0))
                        continue
                    future = others
                    reward = task.reward * row["correct"] if len(row["costs"]) == 1 else 0
                    if len(row["costs"]) > 1:
                        next_state = state.successor(row["state"], row["states"][1], cost)
                        next_f = self.model.predict(
                            e.features, next_state, e.action, e.success_probability
                        )
                        if next_f:
                            future = [*others, (task.reward * next_f.probability, next_f.cost)]
                    outcomes.append((weight, reward + portfolio(future, remaining - cost)))
                q = (
                    sum(w * v for w, v in outcomes)
                    if outcomes
                    else portfolio(
                        [(value(t), forecasts[t.task_id].cost) for t in tasks], remaining
                    )
                )
                # Density breaks near-exact ties in the terminal approximation.
                if (q, density(task)) > (best, density(chosen)):
                    best, chosen = q, task
        f = forecasts[chosen.task_id]
        self.last = {
            "policy": self.mode,
            "predicted_remaining_tokens": round(f.cost, 2),
            "predicted_p80_tokens": round(f.p80, 2),
            "predicted_next_tokens": round(f.next_cost, 2),
            "predicted_remaining_steps": round(f.steps, 2),
            "predicted_success": round(f.probability, 4),
            "state": states[chosen.task_id].as_dict(),
        }
        return chosen
