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


INSTRUCTION_SYSTEM_PROMPT = """You are an instruction extractor. Review the conversation history
and call 'extract_instruction' for each distinct user instruction you find.

Each user message in the conversation represents one instruction from the user.
Extract each one using the extract_instruction tool.

Examples:
- extract_instruction(instruction="explore the project structure")
- extract_instruction(instruction="add tests for the db module")

When you have extracted ALL instructions, call extract_instruction one final time and stop."""


COLLECT_RESULT_SYSTEM_PROMPT = """You are a result aggregator. Review the FULL session history.

Based on the user instructions that were extracted, identify what was accomplished
for each instruction. Focus on WHAT was accomplished, not HOW.

Use bash commands like tool-inspect, grep, head to inspect tool results when needed,
and call 'record_textresult' for each final outcome.

Example:
- record_textresult(desc="Project has 3 core modules: process, state, tool", toolCallLogID=[1,2,5])
- record_textresult(desc="Test suite covers 5 modules with 48 tests", toolCallLogID=[10,12])

When done, respond with only FINISH. Do NOT generate verbose output."""


INSTRUCTION_USER_PROMPT = """Review the conversation history and extract each user instruction
using the extract_instruction tool. Make sure to extract ALL user instructions found in the history."""


COLLECT_RESULT_USER_PROMPT = """Review the FULL session conversation history.
The following user instructions were identified for this session:
{instructions}

For each instruction, identify what was accomplished and record the final outcomes
using record_textresult. Omit intermediate process steps.
Use tool-inspect to verify tool call results when needed.
When done, respond with only FINISH."""


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

        self.wrap_tools()

        agent = Agent(get_api_key=get_api_key)
        agent.set_model(model)
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
            print("abort on instruction extracted")
            self.agent.abort()
            return res
        tool.execute = execute

    def format_result_message(self) -> list[AgentMessage]:
        from pi.ai.types import UserMessage, TextContent

        result: list[AgentMessage] = []

        # 1. UserMessage with aggregated instructions
        instructions = self.commit_data.extracted_instructions
        instructions_text = "\n".join(f"- {i}" for i in instructions) if instructions else "(none)"
        result.append(UserMessage(
            content=[TextContent(text=f"Session instructions:\n{instructions_text}")],
            timestamp=0,
        ))

        # 2. Recorded tool calls and their results
        tool_log_ids: list[int] = []
        for tr in self.commit_data.aggregated_results:
            tool_log_ids.extend(tr.toolCallLogID)
        result.extend(self.tools_mgr.get_all_messages(tool_log_ids))

        # 3. Each TextResult as an individual AssistantMessage
        from pi.ai.types import AssistantMessage
        for tr in self.commit_data.aggregated_results:
            ids = ", ".join(str(i) for i in tr.toolCallLogID) if tr.toolCallLogID else "none"
            result.append(AssistantMessage(
                content=[TextContent(text=f"{tr.desc} [toolCallLogID: {ids}]")],
            ))

        return result

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

    async def _step(self, system_prompt: str, tool_list: list, user_prompt: str):
        self.agent.set_system_prompt(system_prompt)
        self.agent.set_tools(tool_list)
        self.agent.replace_messages(self.message)
        await self.agent.prompt(user_prompt)
        self.message = self.agent.state.messages

    async def process(self, task: Task, context: list[AgentMessage]) -> list[AgentMessage]:
        self.agent.reset()
        self.agent.subscribe(self.on_event)

        index = len(context)
        self.message = context

        # Phase 1: Extract user instructions
        self.instruction_collector.clear()
        phase1_tools = self.tools_mgr.create_all_tools(".")
        phase1_tools.extend(self.instruction_collector.tools)

        await self._step(
            system_prompt=INSTRUCTION_SYSTEM_PROMPT,
            tool_list=phase1_tools,
            user_prompt=INSTRUCTION_USER_PROMPT,
        )

        # Build instructions text for phase 2 prompt
        instructions = self.commit_data.extracted_instructions
        instructions_text = "\n".join(f"- {i}" for i in instructions) if instructions else "(none)"

        # Phase 2: Collect results based on extracted instructions
        self.result_collector.clear()
        phase2_tools = self.tools_mgr.create_all_tools(".")
        phase2_tools.extend(self.result_collector.tools)

        await self._step(
            system_prompt=COLLECT_RESULT_SYSTEM_PROMPT,
            tool_list=phase2_tools,
            user_prompt=COLLECT_RESULT_USER_PROMPT.format(instructions=instructions_text),
        )

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

        return self.format_result_message()
