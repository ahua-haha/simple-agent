"""CommitCollectResultProcess - aggregate results across multiple session runs."""

from __future__ import annotations
import asyncio
from typing import Any

from pi.agent import Agent, AgentToolResult, AgentToolUpdateCallback
from pi.ai import get_model
from pi.agent.types import AgentMessage

from simple_agent.db.db import Database
from simple_agent.globals import TOOL_MGR
from simple_agent.models import register_custom_models, get_api_key
from simple_agent.state.state import (
    EXTRACTED_INSTRUCTION_JSON_SCHEMA,
    TEXT_RESULT_JSON_SCHEMA,
    CommitData,
    ExtractedInstruction,
    Task,
    TextResult,
)
from simple_agent.tool.tool_mgr import ToolMgr
from simple_agent.tool.collector import Collector


SYSTEM_PROMPT = """You are a result aggregator. Review the FULL session history spanning multiple runs.

FIRST: Call 'extract_instruction' for each user instruction you find in the conversation.
THEN: Call 'record_textresult' to record final outcomes across all runs.

Focus on WHAT was accomplished, not HOW. Omit intermediate process details.
Use bash commands like tool-inspect, grep, head to inspect tool results when needed.

Examples:
- extract_instruction(instruction="explore the project structure")
- extract_instruction(instruction="add tests for the db module")
- record_textresult(desc="Project has 3 core modules: process, state, tool", toolCallLogID=[1,2,5])

When done, respond with only FINISH. Do NOT generate verbose output.
"""


class CommitCollectResultProcess:
    agent: Agent
    instruction_collector: Collector
    result_collector: Collector
    tools_mgr: ToolMgr
    message: list[AgentMessage]
    _db: Database

    def __init__(self):
        register_custom_models()
        model = get_model("deepseek", "deepseek-v4-pro")
        self.tools_mgr = TOOL_MGR
        self._db = Database()

        self.instruction_collector = self.tools_mgr.create_collector(
            model_class=ExtractedInstruction,
            name="extract_instruction",
            description="Extract a user instruction from the session history",
            parameters=EXTRACTED_INSTRUCTION_JSON_SCHEMA,
        )

        self.result_collector = self.tools_mgr.create_collector(
            model_class=TextResult,
            name="record_textresult",
            description="Record a TextResult instance capturing a final outcome from the full session",
            parameters=TEXT_RESULT_JSON_SCHEMA,
        )
        self.message = []

        agent = Agent(get_api_key=get_api_key)
        agent.set_model(model)

        bash_tools = self.tools_mgr.create_all_tools(".")
        all_tools = bash_tools
        all_tools.extend(self.instruction_collector.tools)
        all_tools.extend(self.result_collector.tools)

        agent.set_tools(all_tools)
        agent.set_system_prompt(SYSTEM_PROMPT)
        self.agent = agent

    def wrap_tools(self):
        tool = self.instruction_collector.tools[0]
        original = tool.execute
        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            res = await original(tool_call_id, params, cancel_event, on_update)
            if not self.instruction_collector.item:
                return res
            print("pause on instruction extracted")
            self.agent.abort()
            return res
        tool.execute = execute

    @property
    def commit_data(self) -> CommitData:
        instructions: list[str] = []
        if self.instruction_collector.item:
            for item in self.instruction_collector.item:
                if isinstance(item, ExtractedInstruction):
                    instructions.append(item.instruction)

        results: list[TextResult] = []
        if self.result_collector.item:
            results = [item for item in self.result_collector.item if isinstance(item, TextResult)]

        return CommitData(
            extracted_instructions=instructions,
            aggregated_results=results,
        )

    def on_event(self, event):
        """Print events in streaming mode."""
        if event.type == "message_update":
            ae = event.assistant_message_event
            if ae.type == "thinking_start":
                print("<thinking>", end="\n", flush=True)
            if ae.type == "text_start":
                print("<resp>", end="\n", flush=True)
            if ae.type == "thinking_end":
                print("\n</thinking>", end="\n", flush=True)
            if ae.type == "text_end":
                print("\n</resp>", end="\n", flush=True)
            if ae.type == "text_delta":
                print(ae.delta, end="", flush=True)
            elif ae.type == "thinking_delta":
                print(ae.delta, end="", flush=True)
        elif event.type == "tool_execution_start":
            print(f"\n[tool start: {event.tool_name}]", flush=True)
        elif event.type == "tool_execution_end":
            print()
        elif event.type == "agent_end":
            print("\n[agent done]", flush=True)

    async def _step(self, task: Task, system_prompt: str, user_prompt: str):
        self.agent.set_system_prompt(system_prompt)
        self.agent.replace_messages(self.message)
        await self.agent.prompt(user_prompt)
        self.message = self.agent.state.messages

    async def process(self, task: Task, context: list[AgentMessage]) -> list[AgentMessage]:
        self.agent.reset()
        self.agent.subscribe(self.on_event)

        index = len(context)
        self.message = context

        await self._step(task, SYSTEM_PROMPT)

        if self.result_collector.item:
            task.result = [
                item for item in self.result_collector.item if isinstance(item, TextResult)
            ]

        self._db.save_task(
            task_type="commit_collect_result",
            task_input=task.input,
            messages=self.message,
            results=task.result,
            status="finished",
        )

        return self.message[index:]
