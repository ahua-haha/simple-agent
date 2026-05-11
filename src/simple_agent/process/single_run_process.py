"""SingleRunProcess - directly execute tools, then collect results."""

from __future__ import annotations

import asyncio
from typing import Any

from pi.agent import Agent, AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi.ai import UserMessage, TextContent, get_model
from pi.ai.types import AssistantMessage, ToolResultMessage
from pi.agent.types import AgentMessage

from simple_agent.process.collect_result_process import CollectResultProcess
from simple_agent.models import register_custom_models, get_api_key
from simple_agent.state.state import SingleRunTask, StateClarification
from simple_agent.tool.tool_mgr import ToolMgr
from simple_agent.tool.collector import Collector
from simple_agent.db.db import Database
from simple_agent.stream import stream_event


SYSTEM_PROMPT = """You are a helpful assistant. Use the available tools to directly accomplish the user's task.
<important>
When the task is complete and no further tool calls are required, you MUST use 'determine_state' tool to determine the state BEFORE your final response.
</important>

<example>
tool call 1 ...
tool call 1 result ...
tool call 2 ...
tool call 2 result ...

Now the context information is complete. use 'determine_state' tool call to determine the state
Final response: ...
</example>
"""


class SingleRunProcess:
    agent: Agent
    tools_mgr: ToolMgr
    tools: list[AgentTool]
    state_collector: Collector
    message: list[AgentMessage]
    _db: Database

    def __init__(self, tools_mgr: ToolMgr | None = None, db: Database | None = None):
        register_custom_models()
        model = get_model("deepseek", "deepseek-v4-pro")
        self.tools_mgr = tools_mgr or ToolMgr()
        self._db = db or Database()
        self.create_state_clarify_collector()
        self.wrap_tools()

        agent = Agent(get_api_key=get_api_key)
        agent.set_model(model)
        all_tools = self.tools_mgr.create_all_tools(".")
        all_tools.extend(self.state_collector.tools)

        self.tools = all_tools
        self.agent = agent

    def create_state_clarify_collector(self):
        name = "determine_state"
        description = "Determine the current state based on context. States: finished (task complete), error (task failed)"
        tool_schema = {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "description": "Available states:\n- finished: task complete\n- error: task failed",
                    "enum": ["finished", "error"],
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for choosing this state",
                },
            },
            "required": ["state", "reason"],
        }
        self.state_collector = self.tools_mgr.create_collector(StateClarification, name, description, tool_schema)

    def wrap_tools(self):
        tool = self.state_collector.tools[0]
        original = tool.execute
        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            res = await original(tool_call_id, params, cancel_event, on_update)
            if not self.state_collector.item:
                return res
            state = self.state_collector.item[0].state
            print(f"abort on state {state}")
            self.agent.abort()
            return res
        tool.execute = execute

    def prune_message(self):
        lastToolCall = self.message[-2:]
        if isinstance(lastToolCall[0], AssistantMessage) and isinstance(lastToolCall[1], ToolResultMessage) and lastToolCall[1].tool_name == "determine_state":
            print("prune last two determine state tool call")
            del self.message[-2:]

    def format_result_message(self, task: SingleRunTask) -> list[AgentMessage]:
        result = [UserMessage(content=[TextContent(text=task.input)], timestamp=0)]
        tool_log_id = []
        for res in task.result:
            tool_log_id.extend(res.toolCallLogID)

        result.extend(self.tools_mgr.get_all_messages(tool_log_id))

        return result

    async def _step(self, system_prompt: str, tool_list: list, user_prompt: str):
        self.agent.set_system_prompt(system_prompt)
        self.agent.set_tools(tool_list)
        self.agent.replace_messages(self.message)
        await self.agent.prompt(user_prompt)
        self.message = self.agent.state.messages

    async def process(self, task: SingleRunTask, context: list[AgentMessage] = []) -> list[AgentMessage]:
        self.agent.reset()
        self.agent.subscribe(stream_event)

        index = len(context)
        self.message = context

        if task.result is None:
            task.result = []

        await self._step(SYSTEM_PROMPT, self.tools, task.input)
        self.prune_message()

        collectProc = CollectResultProcess(tools_mgr=self.tools_mgr, db=self._db)
        await collectProc.process(task, self.message[index:])

        self._db.save_task(
            task_type="single_run",
            task_input=task.input,
            messages=self.message,
            results=task.result,
            status="finished",
        )

        return self.format_result_message(task)
