import asyncio
import json
import os
from typing import Any

from pi.ai import ToolCall, AssistantMessage, ToolResultMessage
from pi.agent import AgentTool, AgentToolResult, AgentToolUpdateCallback, AgentMessage
from pi.ai.types import TextContent
from pi.coding import create_all_tools

from simple_agent.state.state import ToolExecMessage
from simple_agent.db.db import Database
from simple_agent.snapshot.ghost_indexer import RepoWatcher


def _format(id: int, result: AgentToolResult) -> AgentToolResult:
    orignal_text = result.content[0].text
    new_text = f"<TOOLCALLID>{id}</TOOLCALLID>\n<content>\n{orignal_text}\n</content>"
    result.content[0].text = new_text
    return result

class ToolMgr:
    tools: list[AgentTool]
    records: list[ToolExecMessage]

    def __init__(self, db: Database | None = None):
        self.tools: list[AgentTool] = []
        self._db = db or Database()

    def create_all_tools(self, cwd: str) -> list[AgentTool]:
        tools = list(create_all_tools(cwd).values())
        for tool in tools:
            tool = self.wrap_tools(tool)
        return tools

    def get_all_messages(self, log_id: list[int]) -> list[ToolExecMessage]:
        if not log_id:
            return []
        return self._db.get_tool_calls_by_ids(log_id)



    def wrap_tools(self, tool: AgentTool) -> AgentTool:
        original = tool.execute
        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            res = await original(tool_call_id, params, cancel_event, on_update)
            raw = res.content[0].text
            next_id = self._db.next_tool_call_id()
            res = _format(next_id, res)

            tool_exec = ToolExecMessage(
                tool_call=ToolCall(id=tool_call_id, arguments=params, name=tool.name),
                raw_output=raw,
                tool_result=res,
            )
            self._db.insert_tool_call(tool_exec)
            return res
        tool.execute = execute
        return tool

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
                tool.result = item
                return AgentToolResult(content=[TextContent(text="ok")])
            except Exception as e:
                return AgentToolResult(content=[TextContent(text=f"validation failed: {e}")])

        tool.execute = execute
        tool.result = None
        wrapped_tool = self.wrap_tools(tool)
        return wrapped_tool

    def create_diff_tool(self, repo_watcher: RepoWatcher, start_hash: str, end_hash: str) -> AgentTool:
        tool = AgentTool(
            name="diff",
            description="Show changes between the start and end of the task. Use optional 'path' to diff a single file.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Optional file path to diff a single file"},
                },
                "required": [],
            },
        )

        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            path = params.get("path")
            try:
                if path:
                    output = repo_watcher.get_file_diff(start_hash, end_hash, path)
                else:
                    output = repo_watcher.get_diff(start_hash, end_hash)
                return AgentToolResult(content=[TextContent(text=output or "(no changes)")])
            except Exception as e:
                return AgentToolResult(content=[TextContent(text=f"diff failed: {e}")])

        tool.execute = execute
        return self.wrap_tools(tool)
