from __future__ import annotations

import asyncio
from typing import Any

from pi.ai import ToolCall
from pi.agent import AgentTool, AgentToolResult, AgentToolUpdateCallback

from simple_agent.db.db import Database
from simple_agent.state.state import ToolExecMessage


def _format_tool_result(log_id: int, result: AgentToolResult) -> AgentToolResult:
    original_text = result.content[0].text
    result.content[0].text = (
        f"<TOOLCALLID>{log_id}</TOOLCALLID>\n"
        f"<content>\n{original_text}\n</content>"
    )
    return result


class ToolExecutionLogger:
    """Wrap tools to persist execution records and optionally notify task manager."""

    def __init__(self, db: Database | None = None, task_manager=None):
        self._db = db or Database()
        self._task_manager = task_manager

    def wrap_tool(self, tool: AgentTool) -> AgentTool:
        original = tool.execute

        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            result = await original(tool_call_id, params, cancel_event, on_update)
            raw_output = result.content[0].text
            next_id = self._db.next_tool_call_id()
            result = _format_tool_result(next_id, result)
            tool_exec = ToolExecMessage(
                tool_call=ToolCall(id=tool_call_id, arguments=params, name=tool.name),
                raw_output=raw_output,
                tool_result=result,
            )
            log_id = self._db.insert_tool_call(tool_exec)
            if self._task_manager is not None:
                self._task_manager.record_tool_call(log_id)
            return result

        tool.execute = execute
        return tool

    def wrap_tools(self, tools: list[AgentTool]) -> list[AgentTool]:
        return [self.wrap_tool(tool) for tool in tools]

    def get_all_messages(self, log_ids: list[int]) -> list[ToolExecMessage]:
        if not log_ids:
            return []
        return self._db.get_tool_calls_by_ids(log_ids)
