"""PlanRunner — examine context, create sub-tasks or finish."""

from __future__ import annotations

from typing import TYPE_CHECKING

from simple_agent.process.agent_process import AgentProcess, AgentState
from simple_agent.process.runners import BaseRunner, RunnerResult
from simple_agent.tool.tool_mgr import ToolMgr
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


class PlanRunner(BaseRunner):
    """Runner for plan tasks — create sub-tasks or finish.

    Examines task.context(), runs the agent with the define_task tool.
    If the agent calls it, a sub-task is created.  If the agent responds
    directly (no tool call), the task is finished.
    """

    type = "plan"

    def __init__(self, db: Database, tools_mgr: ToolMgr, agent_process: AgentProcess):
        self._db = db
        self._tools_mgr = tools_mgr
        self._agent_process = agent_process

    async def run(self, task: "Task") -> RunnerResult:
        state = AgentState()
        state.stop_condition = lambda s: "define_task" in s.tool_results
        tools: list = [
            state.bind_tool(self._tools_mgr.create_define_task_tool()),
        ]

        await self._agent_process.run(
            system_prompt=SYSTEM_PROMPT,
            messages=task.metadata["context_msgs"],
            tools=tools,
            state=state,
            user_prompt=task.input,
        )
        task.metadata["context_msgs"].extend(state.new_messages)
        task.messages.extend(state.new_messages)

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

