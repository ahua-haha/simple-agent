"""CommitCollectResultProcess - aggregate results across multiple session runs."""

from __future__ import annotations
import asyncio
from typing import Any

from pi.agent import Agent, AgentToolResult, AgentToolUpdateCallback
from pi.ai import get_model
from pi.agent.types import AgentMessage

from simple_agent.db.db import Database
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
from simple_agent.stream import stream_event


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
    tools_mgr: ToolMgr
    message: list[AgentMessage]
    _db: Database

    def __init__(self, tools_mgr: ToolMgr | None = None, db: Database | None = None):
        register_custom_models()
        model = get_model("deepseek", "deepseek-v4-pro")
        self.tools_mgr = tools_mgr or ToolMgr()
        self._db = db or Database()

        self.instruction_collector = self.tools_mgr.create_record_tool(
            model_class=ExtractedInstruction,
            name="extract_instruction",
            description="Extract a user instruction from the session history",
            parameters=EXTRACTED_INSTRUCTION_JSON_SCHEMA,
        )

        self.result_collector = self.tools_mgr.create_record_tool(
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
        tool = self.instruction_collector
        original = tool.execute
        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            res = await original(tool_call_id, params, cancel_event, on_update)
            if tool.result is None:
                return res
            print("abort on instruction extracted")
            self.agent.abort()
            return res
        tool.execute = execute

    def format_result_message(self) -> list[AgentMessage]:
        from simple_agent.format import format_results
        instructions = self.commit_data.extracted_instructions
        instructions_text = "\n".join(f"- {i}" for i in instructions) if instructions else "(none)"
        task = Task(input="", result=self.commit_data.aggregated_results)
        return format_results(
            self.tools_mgr, task, status="finished",
            label=f"the session\ninstructions:\n{instructions_text}",
        )

    @property
    def commit_data(self) -> CommitData:
        instructions: list[str] = []
        r = self.instruction_collector.result
        if isinstance(r, ExtractedInstruction):
            instructions.append(r.instruction)

        results: list[TextResult] = []
        r2 = self.result_collector.result
        if isinstance(r2, TextResult):
            results.append(r2)

        return CommitData(
            extracted_instructions=instructions,
            aggregated_results=results,
        )

    async def _step(self, system_prompt: str, tool_list: list, user_prompt: str):
        self.agent.set_system_prompt(system_prompt)
        self.agent.set_tools(tool_list)
        self.agent.replace_messages(self.message)
        await self.agent.prompt(user_prompt)
        self.message = self.agent.state.messages

    async def process(self, task: Task, context: list[AgentMessage]) -> list[AgentMessage]:
        self.agent.reset()
        self.agent.subscribe(stream_event)

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
