from __future__ import annotations

import asyncio
from dataclasses import dataclass


class BudgetExhausted(Exception):
    pass


class TaskBudgetExhausted(BudgetExhausted):
    def __init__(self, needed_limit: int):
        super().__init__("Task allocation is too small for another useful call")
        self.needed_limit = needed_limit


@dataclass
class Reservation:
    task_id: str
    units: int
    max_output: int


class Budget:
    """Integer accounting: one unit = 0.1 output token. Cached input counts in full."""

    def __init__(self, limit: int):
        if limit < 0:
            raise ValueError("BUDGET must be nonnegative")
        self.limit = limit * 10
        self.spent = 0
        self.reserved = 0
        self.task_spent: dict[str, int] = {}
        self.task_reserved: dict[str, int] = {}
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.uncertain_units = 0
        self.calls = 0
        self.disabled = False
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)

    @property
    def remaining(self) -> float:
        return max(0, self.limit - self.spent - self.reserved) / 10

    async def reserve(
        self,
        task_id: str,
        prompt_bound: int,
        desired_output: int,
        task_limit: int,
        minimum_output: int = 32,
        wait_for_global: bool = False,
    ) -> Reservation:
        async with self._condition:
            minimum = prompt_bound + minimum_output * 10
            while (
                wait_for_global
                and not self.disabled
                and self.reserved
                and self.limit - self.spent - self.reserved < minimum
                and self.limit - self.spent >= minimum
            ):
                await self._condition.wait()
            global_available = self.limit - self.spent - self.reserved
            if self.disabled or global_available < minimum:
                raise BudgetExhausted("Insufficient global budget for another model call")
            available = min(
                global_available,
                task_limit * 10
                - self.task_spent.get(task_id, 0)
                - self.task_reserved.get(task_id, 0),
            )
            output = min(desired_output, (available - prompt_bound) // 10)
            if output < minimum_output:
                needed = (
                    self.task_spent.get(task_id, 0) + self.task_reserved.get(task_id, 0) + minimum
                )
                raise TaskBudgetExhausted((needed + 9) // 10)
            units = prompt_bound + output * 10
            self.reserved += units
            self.task_reserved[task_id] = self.task_reserved.get(task_id, 0) + units
            return Reservation(task_id, units, output)

    async def settle(self, reservation: Reservation, usage: object | None) -> float:
        async with self._condition:
            self.reserved -= reservation.units
            self.task_reserved[reservation.task_id] -= reservation.units
            self.calls += 1
            if usage is None:
                # A failed/interrupted request may still have generated tokens.
                charge = reservation.units
                self.uncertain_units += charge
            else:
                prompt = max(0, int(usage.prompt_tokens))
                output = max(0, int(usage.completion_tokens))
                self.prompt_tokens += prompt
                self.completion_tokens += output
                # completion_tokens includes reasoning; do not count it twice.
                charge = prompt + 10 * output
            self.spent += charge
            self.task_spent[reservation.task_id] = (
                self.task_spent.get(reservation.task_id, 0) + charge
            )
            self._condition.notify_all()
            return charge / 10

    def report(self) -> dict:
        return {
            "budget": self.limit / 10,
            "spent": self.spent / 10,
            "remaining": self.remaining,
            "reserved": self.reserved / 10,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "uncertain_charge": self.uncertain_units / 10,
        }
