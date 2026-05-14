"""AgentProcess — base class for all agent processes."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from pi.agent import Agent, AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi.ai import get_model
from pi.agent.types import AgentMessage
from pi.ai.types import AssistantMessage, ToolResultMessage

from simple_agent.models import register_custom_models, get_api_key
from simple_agent.tool.collector import Collector


HookFn = Callable[["AgentProcess"], None]


class AgentProcess:
    agent: Agent
    finish_reason: str | None
    message: list[AgentMessage]
    _tools: list[AgentTool]
    _collectors: list[Collector]

    def __init__(self, model):
        register_custom_models()
        self._tools = []
        self._collectors = []
        self.message = []

        agent = Agent(get_api_key=get_api_key)
        agent.set_model(model)
        self.agent = agent

    def stop_agent(self, reason: str):
        self.finish_reason = reason
        self.agent.abort()

    def prune(self, tool_name: str) -> None:
        """Remove the last tool call pair if it matches the given tool name."""
        last_two = self.message[-2:]
        if (
            isinstance(last_two[0], AssistantMessage)
            and isinstance(last_two[1], ToolResultMessage)
            and last_two[1].tool_name == tool_name
        ):
            del self.message[-2:]

    def add_tool(self, tool: AgentTool, on_call: HookFn | None = None) -> None:
        """Add a tool. If on_call is given, it is invoked with self after the tool executes."""
        if on_call:
            original = tool.execute
            async def wrapped(
                tool_call_id: str,
                params: dict[str, Any],
                cancel_event: asyncio.Event | None = None,
                on_update: AgentToolUpdateCallback | None = None,
            ) -> AgentToolResult:
                res = await original(tool_call_id, params, cancel_event, on_update)
                on_call(self)
                return res
            tool.execute = wrapped
        self._tools.append(tool)

    def add_collector(self, collector: Collector, on_call: HookFn | None = None) -> Collector:
        """Add a collector's tools, optionally hook on_call(self) after each tool execution. Returns the collector."""
        for tool in collector.tools:
            self.add_tool(tool, on_call=on_call)
        self._collectors.append(collector)
        return collector
    
    def get_messages(self) -> list[AgentMessage]:
        return self.message

    async def step(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        user_prompt: str,
        tools: list | None = None,
    ) -> str | None:
        """Run the agent, return finish_reason if the agent was stopped, else None."""
        self.finish_reason = None
        self.agent.set_system_prompt(system_prompt)
        self.agent.set_tools(tools if tools is not None else self._tools)
        self.agent.replace_messages(messages)
        await self.agent.prompt(user_prompt)
        self.message = self.agent.state.messages
        return self.finish_reason
