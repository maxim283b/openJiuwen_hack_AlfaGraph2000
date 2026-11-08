from __future__ import annotations

import time
import uuid

from openjiuwen.core.common.logging.log_config import configure_log_config

configure_log_config(
    {
        "level": "ERROR",
        "output": ["console"],
        "interface_output": ["console"],
        "performance_output": ["console"],
    }
)

from openjiuwen.core.session import WORKFLOW_EXECUTE_TIMEOUT  # noqa: E402
from openjiuwen.core.workflow import (  # noqa: E402
    End,
    Start,
    Workflow,
    WorkflowCard,
    WorkflowComponent,
    create_workflow_session,
)


class Stage(WorkflowComponent):
    def __init__(self, action):
        super().__init__()
        self.action = action

    async def invoke(self, inputs, session, context):
        await self.action(inputs["task_id"], inputs["mode"])
        return inputs


class TaskWorkflow:
    """Every task follows a real openJiuwen graph, including validation and submission."""

    def __init__(self, engine):
        self.deadline = engine.deadline
        self.flow = Workflow(
            card=WorkflowCard(
                id="alfagraph-task-" + uuid.uuid4().hex,
                name="Budget-aware task solver",
                version="1.0",
                input_params={
                    "type": "object",
                    "properties": {"task_id": {"type": "string"}, "mode": {"type": "string"}},
                    "required": ["task_id", "mode"],
                },
            )
        )
        self.flow.set_start_comp(
            "start", Start(), inputs_schema={"task_id": "${task_id}", "mode": "${mode}"}
        )
        previous = "start"
        for name, action in [
            ("local", engine.local_stage),
            ("agent", engine.agent_stage),
            ("review", engine.review_stage),
            ("save", engine.save_stage),
        ]:
            self.flow.add_workflow_comp(
                name,
                Stage(action),
                inputs_schema={
                    "task_id": "${" + previous + ".task_id}",
                    "mode": "${" + previous + ".mode}",
                },
            )
            self.flow.add_connection(previous, name)
            previous = name
        self.flow.set_end_comp("end", End(), inputs_schema={"task_id": "${save.task_id}"})
        self.flow.add_connection("save", "end")

    async def run(self, task_id: str, mode: str):
        result = await self.flow.invoke(
            {"task_id": task_id, "mode": mode},
            create_workflow_session(
                session_id=uuid.uuid4().hex,
                envs={WORKFLOW_EXECUTE_TIMEOUT: max(0.01, self.deadline - time.monotonic())},
            ),
        )
        if result.state.value != "COMPLETED":
            raise RuntimeError(f"openJiuwen workflow did not complete: {result.state}")
