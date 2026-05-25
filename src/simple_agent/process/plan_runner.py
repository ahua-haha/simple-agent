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

If enough information has been gathered to answer the original request:
call 'determine_state' with state='finished' and a reason explaining why.

The sub-task agent inherits the full conversation context, so you do not need
to repeat information the sub-task can already see — just describe what to do.
"""


class PlanRunner(BaseRunner):
    """Runner for plan tasks — create sub-tasks or finish.

    Examines task.context(), runs the agent with define_task and
    determine_state tools.  Returns a signal that CentralControl
    uses to move the cursor.
    """

    type = "plan"

    def __init__(self, db: Database, tools_mgr: ToolMgr, agent_process: AgentProcess):
        self._db = db
        self._tools_mgr = tools_mgr
        self._agent_process = agent_process

    async def run(self, task: "Task") -> RunnerResult:
        self._ensure_metadata(task)

        state = AgentState()
        tools: list = [
            state.bind_tool(self._tools_mgr.create_define_task_tool(), stop=True),
            state.bind_tool(self._tools_mgr.create_determine_state_tool(), stop=True),
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

        if "determine_state" in state.tool_results:
            from simple_agent.state.state import StateClarification
            sc = state.tool_results["determine_state"][-1]
            if isinstance(sc, StateClarification) and sc.state == "finished":
                task.result_msg = list(task.messages)
                return RunnerResult(kind="finished")

        return RunnerResult(kind="continue")

    def _ensure_metadata(self, task: "Task") -> None:
        if "context_msgs" not in task.metadata:
            from simple_agent.state.state import Task as TaskModel
            current_id = task.parent_id
            ancestor_rows = []
            while current_id is not None:
                row = self._db.get_task(current_id)
                if row is None:
                    break
                ancestor_rows.append(row)
                current_id = row.get("parent_id")
            ancestor_rows.reverse()
            tasks_by_id = TaskModel.from_db_rows(ancestor_rows) if ancestor_rows else {}
            task.metadata["context_msgs"] = task.context(tasks_by_id) if tasks_by_id else list(task.messages)
