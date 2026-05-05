import asyncio
import json
import os
from typing import Any

from pi.ai import ToolCall, AssistantMessage, ToolResultMessage
from pi.agent import AgentTool, AgentToolResult, AgentToolUpdateCallback, AgentMessage
from pi.ai.types import TextContent
from pi.coding import create_all_tools

from simple_agent.state.state import TextResult, TEXT_RESULT_JSON_SCHEMA, ToolExecMessage
from simple_agent.tool.collector import Collector
from simple_agent.tool.db import Database


def _format(id: int, result: AgentToolResult) -> AgentToolResult:
    orignal_text = result.content[0].text
    new_text = f"<TOOLCALLID>{id}</TOOLCALLID>\n<content>\n{orignal_text}\n</content>"
    result.content[0].text = new_text
    return result

class ToolMgr:
    tools: list[AgentTool]
    records: list[ToolExecMessage]

    def __init__(self):
        self.tools: list[AgentTool] = []
        self._db = Database()

    def create_all_tools(self, cwd: str) -> list[AgentTool]:
        tools = list(create_all_tools(cwd).values())
        for tool in tools:
            tool = self.wrap_tools(tool)
        return tools

    def get_all_messages(self, log_id: list[int]) -> list[AgentMessage]:
        res = []
        if not log_id:
            return res

        records = self._db.get_tool_calls_by_ids(log_id)
        for record in records:
            assistant_msg = AssistantMessage(
                role="assistant",
                content=[record.tool_call],
                stop_reason="tool_use",
            )
            tool_result_msg = ToolResultMessage(
                tool_call_id=record.tool_call.id,
                tool_name=record.tool_call.name,
                content=record.tool_result.content,
                details=record.tool_result.details,
                is_error=False,
            )
            res.extend([assistant_msg, tool_result_msg])
        return res



    def wrap_tools(self, tool: AgentTool) -> AgentTool:
        original = tool.execute
        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            res = await original(tool_call_id, params, cancel_event, on_update)
            raw_output = res.content[0].text

            tool_exec = ToolExecMessage(
                tool_call=ToolCall(id=tool_call_id, arguments=params, name=tool.name),
                raw_output=raw_output,
                tool_result=res
            )
            next_id = self._db.insert_tool_call(tool_exec)
            res = _format(next_id, res)
            return res
        tool.execute = execute
        return tool

    def create_collector(self, model_class: type, name: str, description: str, parameters: dict[str, Any]) -> Collector:
        collector = Collector()
        tool = collector.create_record_tool(
            model_class=model_class,
            name=name,
            description=description,
            parameters=parameters,
        )
        wrapped_tool = self.wrap_tools(tool)
        collector.register_record_tool(wrapped_tool)

        return collector
