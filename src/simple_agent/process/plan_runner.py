"""PlanRunner — examine context, create sub-tasks or finish."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pi.agent import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi.ai.types import TextContent

from simple_agent.process.agent_process import AgentProcess
from simple_agent.process.runners import BaseRunner, RunnerResult
from simple_agent.tool.execution_logger import ToolExecutionLogger
from simple_agent.db.db import Database

if TYPE_CHECKING:
    from simple_agent.state.state import Task

SYSTEM_PROMPT = """You are a planner. Examine the conversation context and decide the next action.

If more exploration or work is needed: call 'define_task' with a clear, self-contained
description of the sub-task to perform.

If enough information has been gathered to answer the original request: respond directly
with a summary of what was accomplished. Do NOT call any tool — just write your response.

The sub-task agent inherits the full conversation context, so you do not need
to repeat information the sub-task can already see — just describe what to do.
"""


class _RecordState(asyncio.Event):
    def __init__(self):
        super().__init__()
        self.tool_results: dict[str, list] = {}
        self.stop_on_tool: str | None = None

    def is_set(self) -> bool:
        if self.stop_on_tool is not None and self.stop_on_tool in self.tool_results:
            return True
        return super().is_set()

    def create_define_task_tool(self) -> AgentTool:
        from simple_agent.state.state import Task

        tool = AgentTool(
            name="define_task",
            description="Define a sub-task to be executed. Include all necessary context.",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "The full input for this sub-task"},
                },
                "required": ["input"],
            },
        )

        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            try:
                item = Task.model_validate(params)
                self.tool_results.setdefault("define_task", []).append(item)
                return AgentToolResult(content=[TextContent(text="ok")])
            except Exception as exc:
                return AgentToolResult(content=[TextContent(text=f"validation failed: {exc}")])

        tool.execute = execute
        return tool


class PlanRunner(BaseRunner):
    """Runner for plan tasks — create sub-tasks or finish.

    Examines task.context(), runs the agent with the define_task tool.
    If the agent calls it, a sub-task is created.  If the agent responds
    directly (no tool call), the task is finished.
    """

    type = "plan"

    def __init__(self, db: Database, execution_logger: ToolExecutionLogger, agent_process: AgentProcess):
        self._db = db
        self._execution_logger = execution_logger
        self._agent_process = agent_process

    async def run(self, task: "Task") -> RunnerResult:
        state = _RecordState()
        state.stop_on_tool = "define_task"
        tools: list = [
            state.create_define_task_tool(),
        ]
        tools = self._execution_logger.wrap_tools(tools)

        new_messages = await self._agent_process.run(
            system_prompt=SYSTEM_PROMPT,
            messages=task.metadata["context_msgs"],
            tools=tools,
            user_prompt=task.input,
            cancel_event=state,
        )
        task.metadata["context_msgs"].extend(new_messages)
        task.messages.extend(new_messages)

        if "define_task" in state.tool_results:
            from simple_agent.state.state import Task as TaskModel
            sub = state.tool_results["define_task"][-1]
            if isinstance(sub, TaskModel):
                sub.type = "explore"
                sub.state = "PENDING"
                sub.result = []
                sub.messages = []
                return RunnerResult(kind="sub_task", child=sub)

        task.result_msg = list(task.messages)
        return RunnerResult(kind="finished")
