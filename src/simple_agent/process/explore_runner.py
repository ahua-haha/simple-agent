"""ExploreRunner — two-phase explore-then-collect workflow."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pi.agent import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi.ai.types import TextContent

from simple_agent.process.agent_process import AgentProcess
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


class _RecordState(asyncio.Event):
    def __init__(self):
        super().__init__()
        self.tool_results: dict[str, list] = {}
        self.stop_on_tool: str | None = None

    def is_set(self) -> bool:
        if self.stop_on_tool is not None and self.stop_on_tool in self.tool_results:
            return True
        return super().is_set()

    def create_record_tool(self, model_class: type, name: str, description: str, parameters: dict[str, Any]) -> AgentTool:
        tool = AgentTool(name=name, description=description, parameters=parameters)

        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            try:
                item = model_class.model_validate(params)
                self.tool_results.setdefault(name, []).append(item)
                return AgentToolResult(content=[TextContent(text="ok")])
            except Exception as exc:
                return AgentToolResult(content=[TextContent(text=f"validation failed: {exc}")])

        tool.execute = execute
        return tool

    def create_determine_state_tool(self) -> AgentTool:
        from simple_agent.state.state import StateClarification

        return self.create_record_tool(
            model_class=StateClarification,
            name="determine_state",
            description="Determine the current state based on context.",
            parameters={
                "type": "object",
                "properties": {
                    "state": {"type": "string", "enum": ["finished", "error"]},
                    "reason": {"type": "string", "description": "Reason for choosing this state"},
                },
                "required": ["state", "reason"],
            },
        )

    def create_record_textresult_tool(self) -> AgentTool:
        from simple_agent.state.state import TEXT_RESULT_JSON_SCHEMA, TextResult

        return self.create_record_tool(
            model_class=TextResult,
            name="record_textresult",
            description="Record a TextResult instance capturing a final outcome.",
            parameters=TEXT_RESULT_JSON_SCHEMA,
        )


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

        state = _RecordState()
        state.stop_on_tool = "determine_state"
        tools: list = [
            state.create_determine_state_tool(),
            *create_all_coding_tools(task.repo_path),
        ]
        tools = self._execution_logger.wrap_tools(tools)
        new_messages = await self._agent_process.run(
            system_prompt=EXECUTE_SYSTEM_PROMPT,
            messages=task.metadata["context_msgs"],
            tools=tools,
            user_prompt=task.input,
            cancel_event=state,
        )
        task.messages.extend(new_messages)

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
        collect_state = _RecordState()
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

        new_messages = await self._agent_process.run(
            system_prompt=COLLECT_SYSTEM_PROMPT,
            messages=task.metadata["context_msgs"],
            tools=collect_tools,
            cancel_event=collect_state,
        )
        task.metadata["context_msgs"].extend(new_messages)
        task.messages.extend(new_messages)

        for tr in collect_state.tool_results.get("record_textresult", []):
            from simple_agent.state.state import TextResult
            if isinstance(tr, TextResult):
                task.result.append(tr)

        task.result_msg = list(task.messages)
        task.state = "FINISHED"
        return RunnerResult(kind="finished")
