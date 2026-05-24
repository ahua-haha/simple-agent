"""ExploreRunner — two-phase explore-then-collect workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from simple_agent.process.agent_process import AgentProcess, AgentState
from simple_agent.process.runners import BaseRunner, RunnerResult
from simple_agent.tool.tool_mgr import ToolMgr
from simple_agent.db.db import Database
from simple_agent.snapshot.ghost_indexer import RepoWatcher

if TYPE_CHECKING:
    from simple_agent.state.state import Task

EXECUTE_SYSTEM_PROMPT = """You are a helpful assistant. Use the available tools to explore and retrieve information.

Important:
- Do NOT generate verbose output — be concise and direct.
- when the task is complete, you MUST IMMEDIATELY call 'determine_state' to clarify the current state.
"""

COLLECT_SYSTEM_PROMPT = """You are a result synthesizer. Review the conversation history,
use bash commands like tool-inspect, grep, sed, head to inspect
tool results, and record each useful outcome as TextResult.

Focus on WHAT was accomplished, not HOW. Each TextResult description MUST:
- Be a single concise sentence stating the outcome
- Mention specific artifacts by name (files, modules, functions, classes)
- Use past tense declarative form: "Found X", "Created Y", "Identified Z"
- Be self-contained — readable without seeing the tool calls

When done, respond with only FINISH. Do NOT generate verbose output.
"""


class ExploreRunner(BaseRunner):
    """Runner for explore tasks — two phases: execute then collect.

    Phase 1: run agent with determine_state + coding tools until finished/failed.
    Phase 2: run agent with record_textresult + coding tools to collect results.
    """

    type = "explore"

    def __init__(self, db: Database, tools_mgr: ToolMgr, agent_process: AgentProcess):
        self._db = db
        self._tools_mgr = tools_mgr
        self._agent_process = agent_process

    async def run(self, task: "Task") -> RunnerResult:
        if task.state == "context_complete":
            return await self._collect(task)

        return await self._execute(task)

    async def _execute(self, task: "Task") -> RunnerResult:
        tasks_by_id = self._load_ancestors(task)
        context_msgs = task.context(tasks_by_id) if tasks_by_id else task.messages

        watcher = RepoWatcher(".", "./data/snapshots")
        task.start_snapshot = task.start_snapshot or watcher.take_snapshot()

        state = AgentState()
        tools: list = [
            state.bind_tool(self._tools_mgr.create_determine_state_tool(), stop=True),
            *self._tools_mgr.create_all_tools("."),
        ]
        await self._agent_process.run(
            system_prompt=EXECUTE_SYSTEM_PROMPT,
            messages=context_msgs,
            tools=tools,
            state=state,
        )
        task.messages.extend(state.new_messages)

        if "determine_state" in state.tool_results:
            from simple_agent.state.state import StateClarification
            sc = state.tool_results["determine_state"][-1]
            if isinstance(sc, StateClarification) and sc.state == "error":
                task.state = "ERROR"
                return RunnerResult(kind="finished")

        task.end_snapshot = watcher.take_snapshot()
        task.state = "context_complete"
        return RunnerResult(kind="continue")

    async def _collect(self, task: "Task") -> RunnerResult:
        tasks_by_id = self._load_ancestors(task)
        context_msgs = task.context(tasks_by_id) if tasks_by_id else task.messages

        watcher = RepoWatcher(".", "./data/snapshots")
        collect_state = AgentState()
        collect_tools: list = [
            collect_state.bind_tool(self._tools_mgr.create_record_textresult_tool()),
        ]
        if task.start_snapshot and task.end_snapshot:
            collect_tools.append(
                self._tools_mgr.create_diff_tool(watcher, task.start_snapshot, task.end_snapshot)
            )
        collect_tools.extend(self._tools_mgr.create_all_tools("."))

        await self._agent_process.run(
            system_prompt=COLLECT_SYSTEM_PROMPT,
            messages=context_msgs,
            tools=collect_tools,
            state=collect_state,
        )
        task.messages.extend(collect_state.new_messages)

        for tr in collect_state.tool_results.get("record_textresult", []):
            from simple_agent.state.state import TextResult
            if isinstance(tr, TextResult):
                task.result.append(tr)

        task.state = "FINISHED"
        return RunnerResult(kind="finished")

    def _load_ancestors(self, task: "Task") -> dict[int, "Task"]:
        """Load ancestor tasks from DB and return id→Task dict for context()."""
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
