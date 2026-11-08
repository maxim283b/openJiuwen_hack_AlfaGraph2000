"""Small empirical cost tables. No model weights, task text, IDs, or answers."""

from __future__ import annotations

import bisect
import json
import math
from pathlib import Path


class CostProfile:
    def __init__(self, data):
        if data.get("schema") != 1 or data.get("unit") != "normalized_output_tokens":
            raise ValueError("Unsupported cost profile schema or token unit")
        self.data = data
        for group in data["types"].values():
            thresholds = group["thresholds"]
            if len(group["buckets"]) != len(thresholds) + 1:
                raise ValueError("Profile buckets do not match thresholds")
            if thresholds != sorted(thresholds) or not all(math.isfinite(x) for x in thresholds):
                raise ValueError("Invalid profile thresholds")
            for cell in [group["overall"], *group["buckets"]]:
                if (
                    cell["attempts"] < 0
                    or not 0 <= cell["correct"] <= cell["attempts"]
                    or not math.isfinite(cell["mean_cost"])
                    or cell["mean_cost"] < 0
                ):
                    raise ValueError("Invalid profile statistics")

    @classmethod
    def load(cls, path):
        if not path:
            return None
        source = Path(path)
        if not source.is_absolute():
            source = Path(__file__).resolve().parents[1] / source
        return cls(json.loads(source.read_text()))

    def predict(self, kind, features, fallback_cost, fallback_probability):
        group = self.data["types"].get(kind)
        if not group or group["overall"]["attempts"] < 5:
            return fallback_cost, fallback_probability
        overall = group["overall"]
        bucket = bisect.bisect_right(group["thresholds"], features.get(group["feature"], 0))
        cell = group["buckets"][min(bucket, len(group["buckets"]) - 1)]
        # Partial pooling prevents a few cheap examples from dominating scheduling.
        local_weight = cell["attempts"] / (cell["attempts"] + 12)
        mean = local_weight * cell["mean_cost"] + (1 - local_weight) * overall["mean_cost"]
        weight = overall["attempts"] / (overall["attempts"] + 10)
        cost = weight * mean + (1 - weight) * fallback_cost
        prior_p = (overall["correct"] + 4 * fallback_probability) / (overall["attempts"] + 4)
        local_p = (cell["correct"] + 12 * prior_p) / (cell["attempts"] + 12)
        probability = weight * local_p + (1 - weight) * fallback_probability
        return max(32, cost), max(0.05, min(0.98, probability))
