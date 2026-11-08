from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

from agentforge_harness.client.llm_client import LLMClient
from agentforge_harness.client.response import StreamEventType
from agentforge_harness.config.config import Config, ModelConfig, ModelProvider
from openai import APIStatusError, AsyncOpenAI

from alfagraph.budget import Budget, BudgetExhausted
from alfagraph.settings import Settings


@dataclass
class Reply:
    text: str = ""
    calls: list = field(default_factory=list)
    finish_reason: str | None = None
    error: str = ""
    cost: float = 0
    malformed_calls: int = 0


class Client:
    """AgentForge serialization/parser with explicit limits and zero hidden SDK retries.

    The 0.1.3 default OpenAI path does not forward max_tokens. The pinned adapter
    deliberately calls its response parser with our bounded request parameters.
    """

    def __init__(self, settings: Settings, budget: Budget, trace):
        self.settings, self.budget, self.trace = settings, budget, trace
        self.client = LLMClient(
            Config(
                model=ModelConfig(
                    provider=ModelProvider.CUSTOM,
                    name=settings.model,
                    base_url=settings.base_url,
                )
            )
        )
        self.client._client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            max_retries=0,
            timeout=settings.request_timeout,
        )
        self.slots = asyncio.Semaphore(settings.concurrency)
        self.rate_lock = asyncio.Lock()
        self.next_request_at = 0.0
        self.cooldown_until = 0.0

    async def complete(
        self,
        task_id: str,
        messages: list,
        tools: list,
        model: str,
        deadline: float,
        *,
        task_limit: int | None = None,
        output_limit: int | None = None,
        minimum_output: int = 32,
    ) -> Reply:
        reply = Reply()
        kwargs = {"model": model, "messages": messages, "stream": False, "temperature": 0}
        if tools:
            kwargs.update(tools=self.client._build_tools(tools), tool_choice="auto")
        if model == "qwen3.8-27b":
            kwargs["reasoning_effort"] = "low"
        # UTF-8 bytes + ample message/schema framing is a conservative input bound.
        prompt_bound = len(json.dumps(kwargs, ensure_ascii=False).encode()) + 1024
        desired = output_limit or (768 if model.endswith("noreason") else 1536)
        async with self.slots:
            async with self.rate_lock:
                now = time.monotonic()
                allowed_at = max(self.next_request_at, self.cooldown_until, now)
                if allowed_at >= deadline:
                    raise BudgetExhausted("No time left for a rate-limited request")
                await asyncio.sleep(max(0, allowed_at - now))
                self.next_request_at = time.monotonic() + 60 / self.settings.rpm
            reservation = await self.budget.reserve(
                task_id,
                prompt_bound,
                desired,
                self.settings.task_tokens if task_limit is None else task_limit,
                minimum_output=minimum_output,
                wait_for_global=True,
            )
            kwargs["max_tokens"] = reservation.max_output
            usage = None
            started = time.monotonic()
            try:
                self.trace(
                    "llm_reserved",
                    task_id=task_id,
                    model=model,
                    reserved=reservation.units / 10,
                    max_output=reservation.max_output,
                )
                async with asyncio.timeout(
                    min(self.settings.request_timeout, max(0.01, deadline - started))
                ):
                    async for event in self.client._non_stream_response(
                        self.client.get_client(), kwargs
                    ):
                        if event.type == StreamEventType.TEXT_DELTA and event.text_delta:
                            reply.text += event.text_delta.content
                        elif event.type == StreamEventType.TOOL_CALL_COMPLETE:
                            reply.calls.append(event.tool_call)
                        elif event.type == StreamEventType.MESSAGE_COMPLETE:
                            usage, reply.finish_reason = event.token_usage, event.finish_reason
                    reply.malformed_calls = sum(
                        not isinstance(call.arguments, dict) or "raw_arguements" in call.arguments
                        for call in reply.calls
                    )
                    if reply.malformed_calls:
                        # Some proxies report tool_calls even when max_tokens cuts JSON.
                        # Never execute partial arguments or guess the missing suffix.
                        reply.finish_reason = "length"
            except APIStatusError as exc:
                reply.error = f"http_{exc.status_code}"
                if exc.status_code == 429:
                    try:
                        retry_after = float(exc.response.headers.get("Retry-After", "2"))
                    except ValueError:
                        retry_after = 2
                    self.cooldown_until = time.monotonic() + max(0, min(60, retry_after))
                if exc.status_code in {401, 402, 403}:
                    self.budget.disabled = True
            except TimeoutError:
                reply.error = "request_timeout"
            except asyncio.CancelledError:
                reply.error = "cancelled"
                raise
            except Exception as exc:
                reply.error = type(exc).__name__
            finally:
                reply.cost = await asyncio.shield(self.budget.settle(reservation, usage))
                self.trace(
                    "llm_call",
                    task_id=task_id,
                    model=model,
                    cost=reply.cost,
                    seconds=round(time.monotonic() - started, 3),
                    error=reply.error,
                    finish_reason=reply.finish_reason,
                    usage_known=usage is not None,
                    malformed_calls=reply.malformed_calls,
                )
        return reply

    async def close(self):
        await self.client.close()
