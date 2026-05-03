import asyncio
from typing import Any

from pi.ai import ToolCall
from pi.agent import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi.coding import create_all_tools


from simple_agent.state.state import ToolExecMessage, TextResult, TEXT_RESULT_JSON_SCHEMA
from simple_agent.tool.collector import Collector


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
        self.records: list[ToolExecMessage] = []

    def create_all_tools(self, cwd: str) -> list[AgentTool]:
        tools = list(create_all_tools(cwd).values())
        for tool in tools:
            tool = self.wrap_tools(tool)
        return tools

    def wrap_tools(self, tool: AgentTool) -> AgentTool:
        original = tool.execute
        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            res = await original(tool_call_id, params, cancel_event, on_update)
            id = len(self.records)
            res = _format(id, res)
            self.records.append(ToolExecMessage(
                input=ToolCall(id=tool_call_id, arguments=params, name=tool.name), output=res
            ))
            return res
        tool.execute = execute
        return tool

    def create_collector(self) -> Collector:
        """Create a Collector for TextResult with a registered tool.

        Returns:
            Collector instance with TextResult record tool registered
        """
        collector = Collector()

        # Create and register the record tool for TextResult
        tool = collector.create_record_tool(
            model_class=TextResult,
            name=f"record_textresult",
            description="Record a TextResult instance with the tool call log ID referencing related tool executions",
            parameters=TEXT_RESULT_JSON_SCHEMA,
        )
        wrapped_tool = self.wrap_tools(tool)
        collector.register_record_tool(wrapped_tool)

        return collector