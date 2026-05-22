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
from simple_agent.index.indexer import AgentIndex


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

    def create_index_tools(self, agent_index: AgentIndex) -> list[AgentTool]:
        tree_tool = AgentTool(
            name="index_tree",
            description="Render the project index as a tree with # descriptions. Use to review what's known about the codebase structure.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Subtree path to render (default: root)"},
                    "depth": {"type": "integer", "description": "Max depth to render (default: unlimited)"},
                },
                "required": [],
            },
        )

        async def tree_execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            output = agent_index.tree(
                path=params.get("path", ""),
                depth=params.get("depth"),
            )
            return AgentToolResult(content=[TextContent(text=output)])

        tree_tool.execute = tree_execute

        update_tool = AgentTool(
            name="index_update",
            description="Add or update an entry in the project index. Use after discovering a new file, class, function, or module.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Entry path, e.g. 'src/main.py' or 'src/main.py:main()'"},
                    "type": {"type": "string", "description": "Entry type: folder, file, class, function, method"},
                    "description": {"type": "string", "description": "Text description of what this entry does"},
                },
                "required": ["path", "type", "description"],
            },
        )

        async def update_execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            agent_index.update(
                path=params["path"],
                type=params.get("type", "file"),
                description=params.get("description", ""),
            )
            return AgentToolResult(content=[TextContent(text="ok")])

        update_tool.execute = update_execute

        remove_tool = AgentTool(
            name="index_remove",
            description="Remove an entry and all its children from the project index.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to remove, e.g. 'src/old_module/'"},
                },
                "required": ["path"],
            },
        )

        async def remove_execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            agent_index.remove(path=params["path"])
            return AgentToolResult(content=[TextContent(text="ok")])

        remove_tool.execute = remove_execute

        return [self.wrap_tools(t) for t in [tree_tool, update_tool, remove_tool]]
