import asyncio
import json
import os
from typing import Any

from pi.ai import ToolCall, AssistantMessage, ToolResultMessage
from pi.agent import AgentTool, AgentToolResult, AgentToolUpdateCallback, AgentMessage
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
        self._next_id: int = 0
        self.load()

    def create_all_tools(self, cwd: str) -> list[AgentTool]:
        tools = list(create_all_tools(cwd).values())
        for tool in tools:
            tool = self.wrap_tools(tool)
        return tools

    def get_all_messages(self, log_id: list[int]) -> list[AgentMessage]:
        res = []
        log_id.sort()
        for id in log_id:
            idx = id - self._next_id
            record = self.records[idx]
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
            id = self._next_id + len(self.records)
            res = _format(id, res)

            self.records.append(ToolExecMessage(
                tool_call=ToolCall(id=tool_call_id, arguments=params, name=tool.name), raw_output=raw_output, tool_result=res
            ))
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

    def flush(self, path: str = "./tool_log.jsonl"):
        """Append records to JSON Lines file and clear in-memory records.

        Args:
            path: Path to the JSON Lines log file
        """
        if not self.records:
            return

        # Ensure directory exists
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Read existing line count to determine starting ID
        start_id = self._next_id

        # Write records as JSON Lines
        with open(path, "a") as f:
            for i, record in enumerate(self.records):
                entry = {
                    "id": start_id + i,
                    "tool": record.tool_call.name,
                    "params": record.tool_call.arguments,
                    "content": record.raw_output,
                }
                f.write(json.dumps(entry) + "\n")

        # Update next_id counter
        self._next_id = start_id + len(self.records)

        # Clear records
        self.records.clear()

    def load(self, path: str = "./tool_log.jsonl"):
        """Load records from JSON Lines file.

        Args:
            path: Path to the JSON Lines log file

        Returns:
            list of records loaded (can be used to reconstruct state if needed)
        """
        records = []
        if not os.path.exists(path):
            return records

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                # Reconstruct AgentToolResult
                tool_call = ToolCall(
                    id=entry.get("params", {}).get("_tool_call_id", ""),
                    arguments=entry.get("params", {}),
                    name=entry.get("tool", ""),
                )
                result = ""
                if entry.get("content"):
                    result = entry["content"]

                records.append(ToolExecMessage(tool_call=tool_call, raw_output=result, tool_result=AgentToolResult()))

                # Track highest ID to maintain counter
                if entry.get("id", 0) >= self._next_id:
                    self._next_id = entry["id"] + 1

        return records