"""ExploreRunner — two-phase explore-then-collect workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from simple_agent.process.agent_process import AgentProcess, AgentState
from simple_agent.process.runners import BaseRunner, RunnerResult
from simple_agent.tool.common_tools import create_all_coding_tools
from simple_agent.tool.execution_logger import ToolExecutionLogger
from simple_agent.db.db import Database
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

    def __init__(self, db: Database, execution_logger: ToolExecutionLogger, agent_process: AgentProcess):
        self._db = db
        self._execution_logger = execution_logger
        self._agent_process = agent_process

    async def run(self, task: "Task") -> RunnerResult:
        if task.state == "context_complete":
            return await self._collect(task)

        return await self._execute(task)

    async def _execute(self, task: "Task") -> RunnerResult:
        watcher = task.metadata["repo_watcher"]
        task.start_snapshot = task.start_snapshot or watcher.take_snapshot()

        state = AgentState()
        state.stop_condition = lambda s: "determine_state" in s.tool_results
        tools: list = [
            state.create_determine_state_tool(),
            *create_all_coding_tools(task.repo_path),
        ]
        tools = self._execution_logger.wrap_tools(tools)
        await self._agent_process.run(
            system_prompt=EXECUTE_SYSTEM_PROMPT,
            messages=task.metadata["context_msgs"],
            tools=tools,
            state=state,
            user_prompt=task.input,
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
        collect_state = AgentState()
        collect_tools: list = [
            collect_state.create_record_textresult_tool(),
        ]
        if task.start_snapshot and task.end_snapshot:
            watcher = task.metadata["repo_watcher"]
            collect_tools.append(
                watcher.create_diff_tool(task.start_snapshot, task.end_snapshot)
            )
        collect_tools.extend(create_all_coding_tools(task.repo_path))
        collect_tools = self._execution_logger.wrap_tools(collect_tools)

        await self._agent_process.run(
            system_prompt=COLLECT_SYSTEM_PROMPT,
            messages=task.metadata["context_msgs"],
            tools=collect_tools,
            state=collect_state,
        )
        task.metadata["context_msgs"].extend(collect_state.new_messages)
        task.messages.extend(collect_state.new_messages)

        for tr in collect_state.tool_results.get("record_textresult", []):
            from simple_agent.state.state import TextResult
            if isinstance(tr, TextResult):
                task.result.append(tr)

        task.result_msg = list(task.messages)
        task.state = "FINISHED"
        return RunnerResult(kind="finished")
