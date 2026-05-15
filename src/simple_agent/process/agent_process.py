"""AgentProcess — composable single-step agent execution with chainable API."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from pi.agent import Agent, AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi.ai import get_model
from pi.agent.types import AgentMessage
from pi.ai.types import AssistantMessage, ToolResultMessage

from simple_agent.models import register_custom_models, get_api_key


HookFn = Callable[["AgentProcess"], None]


class AgentProcess:
    agent: Agent
    finish_reason: str | None
    message: list[AgentMessage]
    _results: dict[str, Any]
    _tools: list[AgentTool]

    def __init__(self, model):
        register_custom_models()
        self._tools = []
        self._results: dict[str, Any] = {}
        self.message = []
        self.finish_reason = None

        agent = Agent(get_api_key=get_api_key)
        agent.set_model(model)
        self.agent = agent

    def stop_agent(self, reason: str):
        self.finish_reason = reason
        self.agent.abort()

    def add_tool(self, tool: AgentTool, on_call: HookFn | None = None, store: bool = False) -> AgentProcess:
        """Add a tool. If on_call is given, the tool's .result is captured to self._results[tool.name]
        and on_call(self) is invoked after execution. Returns self."""
        if on_call:
            original = tool.execute
            async def wrapped(
                tool_call_id: str,
                params: dict[str, Any],
                cancel_event: asyncio.Event | None = None,
                on_update: AgentToolUpdateCallback | None = None,
            ) -> AgentToolResult:
                res = await original(tool_call_id, params, cancel_event, on_update)
                if store:
                    self._results[tool.name] = tool.result
                    tool.result = None
                on_call(self)
                return res
            tool.execute = wrapped
        self._tools.append(tool)
        return self

    async def step(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        user_prompt: str,
        tools: list | None = None,
    ) -> AgentProcess:
        """Run the agent. Returns self for chaining."""
        self.finish_reason = None
        self.agent.set_system_prompt(system_prompt)
        self.agent.set_tools(tools if tools is not None else self._tools)
        self.agent.replace_messages(messages)
        await self.agent.prompt(user_prompt)
        self.message = self.agent.state.messages
        return self

    def prune(self, tool_name: str) -> AgentProcess:
        """Remove the last tool call pair if it matches tool_name. Returns self for chaining."""
        last_two = self.message[-2:]
        if (
            isinstance(last_two[0], AssistantMessage)
            and isinstance(last_two[1], ToolResultMessage)
            and last_two[1].tool_name == tool_name
        ):
            del self.message[-2:]
        return self

    def result(self) -> tuple:
        """Return the recorded result for the named tool, or None."""
        res = (self.message, self.finish_reason, self._results)
        self.message = []
        self.finish_reason = None
        self._results = None
        return res
