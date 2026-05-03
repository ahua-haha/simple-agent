
import asyncio
from typing import Any

from pi.ai import ToolCall
from pi.agent import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi.coding import create_all_tools


from simple_agent.state.state import ToolExecMessage

def _format(id: int, result: AgentToolResult) -> AgentToolResult:
    orignal_text = result.content[0].text
    new_text = f"<TOOLCALLID>{id}</TOOLCALLID>\n<content>\n{orignal_text}\n</content>"
    result.content[0].text = new_text
    return result

class ToolMgr:
    tools: list[AgentTool]
    records: list[ToolExecMessage]

    def __init__(self):
        self.records = list()

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
                ToolCall(id=tool_call_id, arguments=params, name=tool.name), res
            ))
            return res
        tool.execute = execute
        return tool