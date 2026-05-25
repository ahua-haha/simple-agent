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
        tasks_by_id = self._load_ancestors(task)
        context_msgs = task.context(tasks_by_id) if tasks_by_id else task.messages

        state = AgentState()
        tools: list = [
            state.bind_tool(self._tools_mgr.create_define_task_tool(), stop=True),
            state.bind_tool(self._tools_mgr.create_determine_state_tool(), stop=True),
        ]

        await self._agent_process.run(
            system_prompt=SYSTEM_PROMPT,
            messages=context_msgs,
            tools=tools,
            state=state,
            user_prompt=task.input,
        )
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
                return RunnerResult(kind="finished")

        return RunnerResult(kind="continue")

    def _load_ancestors(self, task: "Task") -> dict[int, "Task"]:
        from simple_agent.state.state import Task as TaskModel

        current_id = task.parent_id
        ancestor_rows = []
        while current_id is not None:
            row = self._db.get_task(current_id)
            if row is None:
                break
            ancestor_rows.append(row)
            current_id = row.get("parent_id")

        if not ancestor_rows:
            return {}

        ancestor_rows.reverse()
        return TaskModel.from_db_rows(ancestor_rows)
