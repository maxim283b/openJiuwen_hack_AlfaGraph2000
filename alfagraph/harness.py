from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from alfagraph.agent import AgentSession, solve_with_agent
from alfagraph.allocation import Allocation, Progress
from alfagraph.bandit import Arm, LinUCB, resolve_config
from alfagraph.budget import Budget
from alfagraph.client import Client
from alfagraph.execution import commit_sources, prepare, run_tests
from alfagraph.forecast import ForecastPolicy, FutureModel
from alfagraph.local import InsufficientData, solve_csv, solve_math, solve_search
from alfagraph.planner import Planner
from alfagraph.profile import CostProfile
from alfagraph.repair import repair
from alfagraph.review import Reviewer
from alfagraph.review_policy import ReviewPolicy
from alfagraph.selection import SelectionPolicy
from alfagraph.tasks import atomic_write, load_tasks, normalize_answer
from alfagraph.workflow import TaskWorkflow


class Harness:
    def __init__(
        self,
        settings,
        tasks_dir: Path,
        scratch: Path,
        report: Path,
        traces: Path,
        resume=False,
        offline=False,
    ):
        self.settings, self.scratch, self.report_path, self.traces_path = (
            settings,
            scratch,
            report,
            traces,
        )
        self.tasks = load_tasks(tasks_dir)
        self.budget, self.planner = (
            Budget(settings.budget),
            Planner(
                CostProfile.load(settings.profile),
                LinUCB.load(settings.bandit),
                settings.task_tokens,
                SelectionPolicy.load(settings.selection, settings.bandit),
                settings.budget,
            ),
        )
        self.assignments = {}
        self.sessions = {}
        self.resumed_steps = {}
        self.future = (
            ForecastPolicy(
                FutureModel.load(settings.forecast, settings.bandit), settings.forecast_policy
            )
            if settings.forecast
            else None
        )
        if self.future and (settings.selection or settings.assignments):
            raise ValueError(
                "Forecast planning cannot be combined with selection or exploration assignments"
            )
        if settings.assignments:
            assignment_data = json.loads(resolve_config(settings.assignments).read_text())
            self.assignments = {tid: Arm(**arm) for tid, arm in assignment_data.items()}
            if set(self.assignments) != {task.task_id for task in self.tasks}:
                raise ValueError("Exploration assignments must cover the exact input task set")
        self.started = time.monotonic()
        self.deadline = self.started + settings.time_limit
        self.attempts = {}
        self.results = {}
        self.allocations = {}
        self.client = None
        self.reviewer = None
        self.review_policy = ReviewPolicy.load(settings.review_policy)
        self.review_states = {}
        review_state = {}
        self.offline = offline or not settings.api_key
        if resume:
            if not report.is_file():
                raise ValueError("--resume requires the previous --report to preserve budget usage")
            previous = json.loads(report.read_text())
            review_state = previous.get("review_state", {})
            self.resumed_steps = {
                tid: value["steps"] for tid, value in previous.get("agent_sessions", {}).items()
            }
            state = previous.get("selection_state", {})
            self.planner.feedback = state.get("feedback", {})
            self.planner.initial_pending = state.get("initial_pending")
            budget = previous["accounting"]
            outstanding = round(budget.get("reserved", 0) * 10)
            self.budget.spent = round(budget["spent"] * 10) + outstanding
            self.budget.prompt_tokens = budget["prompt_tokens"]
            self.budget.completion_tokens = budget["completion_tokens"]
            self.budget.uncertain_units = round(budget["uncertain_charge"] * 10) + outstanding
            self.budget.calls = budget["calls"]
            self.budget.task_spent = previous.get("task_spent_units", {})
            for task_id, saved in previous.get("allocations", {}).items():
                # Progress evidence belonged to a disposable workspace and is
                # deliberately not restored. Previously charged tokens remain charged.
                hard = min(settings.task_tokens, int(saved["hard_limit"]))
                self.allocations[task_id] = Allocation(
                    settings.allocation,
                    saved["predicted"],
                    min(hard, saved["limit"]),
                    hard,
                    settings.progress_patience,
                    stage=saved["stage"],
                    route_model=saved.get("route_model"),
                    route_action=saved.get("route_action"),
                )
        for task in self.tasks:
            answer_path = task.directory / "answer.txt"
            if answer_path.is_file():
                if not resume:
                    raise ValueError(
                        "Existing answer.txt found; use --resume with its previous report"
                    )
                answer = normalize_answer(task, answer_path.read_text())
                self.results[task.task_id] = {
                    "status": "existing",
                    "answer": answer,
                    "reward": task.reward,
                }
        if not self.offline:
            self.client = Client(settings, self.budget, self.trace)
            if settings.verification:
                self.reviewer = Reviewer(
                    self.client,
                    settings,
                    self.deadline,
                    review_state.get("limits"),
                    review_state.get("calls"),
                )

    def trace(self, event, **data):
        record = {"event": event, "elapsed": round(time.monotonic() - self.started, 3), **data}
        self.traces_path.parent.mkdir(parents=True, exist_ok=True)
        with self.traces_path.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
        if event in {"llm_reserved", "llm_call", "allocation_started", "allocation_extended"}:
            self.save_report()

    def save_report(self):
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": 1,
            "framework": "openjiuwen==0.1.19",
            "elapsed": round(time.monotonic() - self.started, 3),
            "accounting": self.budget.report(),
            "tasks": self.results,
            "task_spent_units": self.budget.task_spent,
            "allocation_mode": self.settings.allocation,
            "forecast_policy": self.settings.forecast_policy,
            "verification_enabled": self.settings.verification,
            "review_state": {"limits": self.reviewer.limits, "calls": self.reviewer.calls}
            if self.reviewer
            else {},
            "agent_states": {tid: s.state.as_dict() for tid, s in self.sessions.items()},
            "agent_sessions": {
                **{
                    tid: {"steps": steps, "finished": False, "context_groups": 0}
                    for tid, steps in self.resumed_steps.items()
                },
                **{
                    tid: {"steps": s.steps, "finished": s.finished, "context_groups": len(s.groups)}
                    for tid, s in self.sessions.items()
                },
            },
            "selection_state": {
                "feedback": self.planner.feedback,
                "initial_pending": self.planner.initial_pending,
            },
            "allocations": {tid: item.report() for tid, item in self.allocations.items()},
            "solved": sum(r["status"] in {"solved", "existing"} for r in self.results.values()),
            "submitted_reward": sum(
                r["reward"] for r in self.results.values() if r["status"] in {"solved", "existing"}
            ),
        }
        atomic_write(self.report_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    async def local_stage(self, task_id, mode):
        attempt = self.attempts[task_id]
        if attempt.local_tried or attempt.status != "pending":
            return
        attempt.local_tried = True
        task = attempt.task
        try:
            if task.kind == "math":
                attempt.answer = solve_math(task.instruction)
            elif task.kind == "search":
                attempt.answer = solve_search(task.instruction, attempt.work)
            elif task.kind == "data_analysis":
                attempt.answer = solve_csv(task, attempt.work)
            elif task.kind == "coding":
                try:
                    async with asyncio.timeout(
                        min(20, max(0.01, self.deadline - time.monotonic()))
                    ):
                        if await repair(attempt):
                            attempt.answer = "DONE"
                except TimeoutError:
                    attempt.reason = "Local repair time limit"
            if attempt.answer is not None:
                attempt.method = "deterministic"
        except InsufficientData as exc:
            attempt.status, attempt.reason = "unsolvable", str(exc)
        except (ValueError, OSError) as exc:
            attempt.reason = type(exc).__name__

    async def agent_stage(self, task_id, mode):
        attempt = self.attempts[task_id]
        if (
            mode != "agent"
            or attempt.answer is not None
            or attempt.status != "pending"
            or self.client is None
        ):
            return
        if task_id not in self.allocations:
            estimate = self.planner.estimate(attempt.task)
            self.allocations[task_id] = Allocation.create(
                estimate,
                self.settings.task_tokens,
                self.settings.allocation,
                self.settings.progress_patience,
            )
            action = self.assignments.get(task_id) or estimate.action
            if action:
                allocation = self.allocations[task_id]
                allocation.hard_limit = min(self.settings.task_tokens, action.tokens)
                # An arm is a full per-task allowance; actual calls still reserve
                # globally and unused capacity is never charged.
                allocation.limit = allocation.hard_limit
                allocation.route_model, allocation.route_action = action.model, action.name
        allocation = self.allocations[task_id]
        first_step = task_id not in self.sessions
        session = self.sessions.setdefault(task_id, AgentSession()) if self.future else None
        if session and first_step:
            session.steps = self.resumed_steps.get(task_id, 0)
            session.state.spent = self.budget.task_spent.get(task_id, 0) / 10
        progress = session.progress if session else Progress()
        if attempt.task.kind == "coding" and (not session or first_step):
            attempt.test_result = await run_tests(attempt)
            progress.tests(attempt.test_result, baseline=True)
        self.trace("allocation_started", task_id=task_id, **allocation.report())
        await solve_with_agent(
            attempt,
            self.client,
            self.settings,
            self.deadline,
            allocation,
            progress,
            session=session,
            one_step=bool(self.future),
        )

    async def save_stage(self, task_id, mode):
        attempt = self.attempts[task_id]
        if attempt.status == "solved":
            return
        if self.reviewer and mode == "agent" and not attempt.review_done:
            session = self.sessions.get(task_id)
            if (
                attempt.answer is not None
                or attempt.status == "unsolvable"
                or not session
                or session.finished
            ):
                attempt.review_pending = True
                self.results[task_id] = {
                    "status": "awaiting_review",
                    "answer": None,
                    "reward": attempt.task.reward,
                    "reason": "Independent check pending",
                    "cost": self.budget.task_spent.get(task_id, 0) / 10,
                }
                self.save_report()
                return
        if attempt.answer is not None:
            answer = normalize_answer(attempt.task, attempt.answer)
            if attempt.task.kind == "coding":
                tests = await run_tests(attempt)
                if not tests["ok"]:
                    attempt.answer = None
                    attempt.reason = "Final test validation failed"
                    return
                commit_sources(attempt)
            atomic_write(attempt.task.directory / "answer.txt", answer + "\n")
            attempt.status = "solved"
        self.results[task_id] = {
            "status": attempt.status,
            "answer": attempt.answer,
            "method": attempt.method,
            "reason": attempt.reason,
            "reward": attempt.task.reward,
            "cost": self.budget.task_spent.get(task_id, 0) / 10,
            "verification": attempt.verification,
        }
        self.trace("task_result", task_id=task_id, **self.results[task_id])
        self.save_report()

    async def review_stage(self, task_id, mode):
        if mode == "review" and self.reviewer:
            await self.reviewer.review(self.attempts[task_id])

    async def select_reviews(self, checks):
        """Allocate only the remaining budget; skipped candidates retain their answers."""
        states = {
            **self.review_states,
            **{tid: session.state.as_dict() for tid, session in self.sessions.items()},
        }
        for tid, allocation in self.allocations.items():
            states.setdefault(tid, {}).update(
                producer_model=allocation.route_model or self.settings.model,
                task_limit=allocation.hard_limit,
            )
        # Unselected candidates are immutable during this phase. Parse their files once.
        predictions = {
            t.task_id: self.review_policy.predict(
                self.attempts[t.task_id],
                self.budget.task_spent.get(t.task_id, 0) / 10,
                states.get(t.task_id, {}),
            )
            for t in checks
        }
        while checks and time.monotonic() < self.deadline:
            batch, remaining = [], self.budget.remaining
            for _ in range(self.settings.concurrency):
                choice = self.review_policy.choose(
                    [self.attempts[t.task_id] for t in checks],
                    remaining,
                    self.budget.task_spent,
                    states,
                    task_ceiling=self.settings.task_tokens,
                    predictions=predictions,
                )
                if choice is None:
                    break
                task, prediction = choice
                checks.remove(task)
                remaining -= prediction["cost"]
                self.trace("review_selected", task_id=task.task_id, **prediction)
                batch.append(task)
            if not batch:
                break
            await asyncio.gather(*(self.run_one(t, "review") for t in batch))
        for task in checks:
            a = self.attempts[task.task_id]
            a.review_pending, a.review_done = False, True
            a.verification = {
                "status": "not_selected",
                "cost": 0,
                "calls": 0,
                "reason": "Predicted gain/cost or remaining budget",
            }
            await self.save_stage(task.task_id, "review")

    async def run_one(self, task, mode):
        original = self.attempts[task.task_id].answer
        before = self.budget.task_spent.get(task.task_id, 0)
        try:
            async with asyncio.timeout(max(0.01, self.deadline - time.monotonic())):
                # Workflow graphs contain mutable execution state. Never share a
                # graph instance between concurrently executing sessions.
                await TaskWorkflow(self).run(task.task_id, mode)
        except TimeoutError:
            self.attempts[task.task_id].reason = "Time limit reached"
            if task.task_id in self.sessions:
                self.sessions[task.task_id].finished = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            attempt = self.attempts[task.task_id]
            attempt.reason = type(exc).__name__
            if task.task_id in self.sessions:
                self.sessions[task.task_id].finished = True
            self.trace("task_error", task_id=task.task_id, error=type(exc).__name__)
        finally:
            attempt = self.attempts[task.task_id]
            if mode == "review" and not attempt.review_done:
                attempt.answer = original
                attempt.review_pending, attempt.review_done = False, True
                attempt.verification = {
                    "status": "unreviewed",
                    "cost": (self.budget.task_spent.get(task.task_id, 0) - before) / 10,
                    "calls": self.reviewer.calls.get(task.task_id, 0) if self.reviewer else 0,
                    "reason": "Review interrupted; original candidate retained",
                }
                await self.save_stage(task.task_id, "review")
            if attempt.review_pending:
                self.results[task.task_id] = {
                    "status": "awaiting_review",
                    "answer": None,
                    "method": attempt.method,
                    "reason": attempt.reason,
                    "reward": task.reward,
                    "cost": self.budget.task_spent.get(task.task_id, 0) / 10,
                }
            elif attempt.status != "solved":
                self.results[task.task_id] = {
                    "status": attempt.status,
                    "answer": None,
                    "method": attempt.method,
                    "reason": attempt.reason,
                    "reward": task.reward,
                    "cost": self.budget.task_spent.get(task.task_id, 0) / 10,
                    "verification": attempt.verification,
                }
            self.save_report()

    async def run(self):
        try:
            # Deterministic work is prioritized independently of LLM budget.
            for task in sorted(self.tasks, key=lambda t: (-t.reward, t.task_id)):
                if task.task_id in self.results or time.monotonic() >= self.deadline:
                    continue
                try:
                    self.attempts[task.task_id] = prepare(task, self.scratch)
                except (ValueError, OSError) as exc:
                    self.results[task.task_id] = {
                        "status": "invalid_input",
                        "reward": task.reward,
                        "reason": type(exc).__name__,
                        "answer": None,
                    }
                    continue
                await self.run_one(task, "local")
            pending = [
                a.task for a in self.attempts.values() if a.status == "pending" and a.answer is None
            ]
            while pending and not self.offline and time.monotonic() < self.deadline:
                # Keep a substantial share available for independent checks.
                if self.reviewer and self.budget.spent >= self.budget.limit * 0.65:
                    break
                batch = []
                planned = 0.0
                for _ in range(self.settings.concurrency):
                    task = (
                        min(pending, key=lambda task: task.task_id)
                        if self.assignments and pending
                        else self.future.choose(
                            pending,
                            max(0, self.budget.remaining - planned),
                            self.planner,
                            self.sessions,
                        )
                        if self.future
                        else self.planner.choose(pending, self.budget.remaining)
                    )
                    if task is None or self.budget.disabled:
                        break
                    if self.future:
                        planned += self.future.last["predicted_next_tokens"]
                    pending.remove(task)
                    estimate = self.planner.estimate(task)
                    self.trace(
                        "task_selected",
                        task_id=task.task_id,
                        estimated_cost=round(estimate.cost, 2),
                        allocation_forecast=round(estimate.tokens, 2),
                        probability=round(estimate.success_probability, 3),
                        priority=estimate.priority(task.reward),
                        features=estimate.features,
                        bandit_action=estimate.action.as_dict() if estimate.action else None,
                        forecast=self.future.last if self.future else None,
                    )
                    batch.append(task)
                if not batch:
                    break
                await asyncio.gather(*(self.run_one(task, "agent") for task in batch))
                for task in batch:
                    attempt = self.attempts[task.task_id]
                    if self.future:
                        session = self.sessions.get(task.task_id)
                        if (
                            session
                            and not session.finished
                            and attempt.status == "pending"
                            and attempt.answer is None
                        ):
                            pending.append(task)
                            continue
                    self.planner.record(
                        task,
                        self.budget.task_spent.get(task.task_id, 0) / 10,
                        attempt.status == "solved" or attempt.answer is not None,
                    )
            if self.reviewer:
                # Scheduling is based on verification risk and public reward,
                # never on hidden correctness or the benchmark's impossible flag.
                priority = {"search": 4, "coding": 3, "data_analysis": 2, "math": 1}
                checks = [
                    a.task
                    for a in self.attempts.values()
                    if a.status != "solved" and not a.review_done and a.task.reward > 0
                ]
                checks.sort(key=lambda t: (priority[t.kind], t.reward), reverse=True)
                if self.review_policy:
                    await self.select_reviews(checks)
                    checks = []
                for index in range(0, len(checks), self.settings.concurrency):
                    if time.monotonic() >= self.deadline:
                        break
                    await asyncio.gather(
                        *(
                            self.run_one(task, "review")
                            for task in checks[index : index + self.settings.concurrency]
                        )
                    )
            for task in self.tasks:
                if task.task_id not in self.results:
                    self.results[task.task_id] = {
                        "status": "not_attempted",
                        "reward": task.reward,
                        "answer": None,
                    }
            self.save_report()
        finally:
            if self.client:
                await self.client.close()
            self.save_report()
        return json.loads(self.report_path.read_text())
