from __future__ import annotations

import os
from dataclasses import dataclass

MODELS = {"qwen3.6-35b-a3b", "qwen3.6-35b-a3b-noreason", "qwen3.8-27b", "qwen3.8-27b-noreason"}


@dataclass
class Settings:
    budget: int = 100_000
    base_url: str = "https://api.neuraldeep.ru/v1"
    api_key: str = ""
    concurrency: int = 3
    time_limit: float = 3500
    model: str = "qwen3.6-35b-a3b-noreason"
    fallback_model: str = "qwen3.8-27b"
    max_steps: int = 8
    task_tokens: int = 6000
    request_timeout: float = 45
    rpm: int = 120
    allocation: str = "adaptive"
    progress_patience: int = 3
    profile: str = ""
    bandit: str = ""
    assignments: str = ""
    selection: str = ""
    forecast: str = ""
    forecast_policy: str = ""
    verification: bool = False
    review_policy: str = ""
    review_strategy: str = "learned"
    review_tokens: int = 3200
    review_model: str = "qwen3.8-27b"
    review_fast_model: str = "qwen3.6-35b-a3b-noreason"

    @classmethod
    def from_env(cls):
        settings = cls(
            budget=int(os.getenv("BUDGET", "100000")),
            base_url=os.getenv("LLM_BASE_URL", cls.base_url).rstrip("/"),
            api_key=os.getenv("LLM_API_KEY", ""),
            concurrency=int(os.getenv("AGENT_CONCURRENCY", "3")),
            time_limit=float(os.getenv("AGENT_TIME_LIMIT", "3500")),
            model=os.getenv("AGENT_MODEL", cls.model),
            fallback_model=os.getenv("AGENT_FALLBACK_MODEL", cls.fallback_model),
            max_steps=int(os.getenv("AGENT_MAX_STEPS", "8")),
            task_tokens=int(os.getenv("AGENT_TASK_TOKENS", "6000")),
            rpm=int(os.getenv("AGENT_RPM", "120")),
            allocation=os.getenv("AGENT_ALLOCATION", "adaptive"),
            progress_patience=int(os.getenv("AGENT_PROGRESS_PATIENCE", "3")),
            profile=os.getenv("AGENT_PROFILE", ""),
            bandit=os.getenv("AGENT_BANDIT", ""),
            assignments=os.getenv("AGENT_ASSIGNMENTS", ""),
            selection=os.getenv("AGENT_SELECTION_POLICY", ""),
            forecast=os.getenv("AGENT_FORECAST_PROFILE", ""),
            forecast_policy=os.getenv("AGENT_FORECAST_POLICY", ""),
            verification=os.getenv("AGENT_VERIFICATION", "0") == "1",
            review_policy=os.getenv("AGENT_REVIEW_POLICY", ""),
            review_strategy=os.getenv("AGENT_REVIEW_STRATEGY", "learned"),
            review_tokens=int(os.getenv("AGENT_REVIEW_TOKENS", "3200")),
            review_model=os.getenv("AGENT_REVIEW_MODEL", cls.review_model),
            review_fast_model=os.getenv("AGENT_REVIEW_FAST_MODEL", cls.review_fast_model),
        )
        if (
            settings.model not in MODELS
            or settings.fallback_model not in MODELS
            or not 1 <= settings.concurrency <= 5
            or settings.max_steps < 1
            or settings.budget < 0
            or settings.task_tokens < 0
            or settings.time_limit <= 0
            or settings.rpm < 1
            or settings.allocation not in {"adaptive", "fixed"}
            or settings.progress_patience < 1
            or settings.forecast_policy not in {"", "step", "density", "knapsack", "mpc"}
            or bool(settings.forecast) != bool(settings.forecast_policy)
            or settings.review_tokens < 512
            or settings.review_strategy not in {"learned", "search_only"}
            or settings.review_model not in MODELS
            or settings.review_fast_model not in MODELS
        ):
            raise ValueError("Invalid model, budget, concurrency, step or time limit setting")
        return settings
