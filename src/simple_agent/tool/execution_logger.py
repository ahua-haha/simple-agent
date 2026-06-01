from __future__ import annotations

import asyncio
import time
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


def _tool_result_payload(result: AgentToolResult) -> dict[str, Any]:
    return {
        "content": [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item.__dict__)
            for item in result.content
        ],
        "details": result.details,
    }


class ToolExecutionLogger:
    """Wrap tools to persist execution records and optionally notify task manager."""

    def __init__(self, db: Database | None = None, task_manager=None, session_id: str | None = None):
        self._db = db or Database()
        self._task_manager = task_manager
        self._session_id = session_id

    def wrap_tool(self, tool: AgentTool) -> AgentTool:
        original = tool.execute

        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            started_at = time.time()
            try:
                result = await original(tool_call_id, params, cancel_event, on_update)
            except Exception as exc:
                if self._session_id is not None:
                    self._db.insert_runner_tool_call(
                        session_id=self._session_id,
                        tool_call_id=tool_call_id,
                        tool_name=tool.name,
                        params=params,
                        result=None,
                        status="error",
                        started_at=started_at,
                        finished_at=time.time(),
                        error=str(exc),
                    )
                raise

            raw_output = result.content[0].text
            if self._session_id is not None:
                self._db.insert_runner_tool_call(
                    session_id=self._session_id,
                    tool_call_id=tool_call_id,
                    tool_name=tool.name,
                    params=params,
                    result=_tool_result_payload(result),
                    status="success",
                    started_at=started_at,
                    finished_at=time.time(),
                    error=None,
                )

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
