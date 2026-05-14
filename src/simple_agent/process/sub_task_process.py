"""SubTaskProcess - planner agent that decomposes tasks into sub-tasks."""

from __future__ import annotations

import asyncio
from typing import Any

from pi.agent import Agent, AgentToolResult, AgentToolUpdateCallback
from pi.ai import UserMessage, TextContent, get_model
from pi.ai.types import AssistantMessage, ToolResultMessage
from pi.agent.types import AgentMessage, AgentTool

from simple_agent.process.explore_process import ExploreProcess
from simple_agent.process.collect_result_process import CollectResultProcess
from simple_agent.models import register_custom_models, get_api_key
from simple_agent.state.state import Task, StateClarification
from simple_agent.tool.tool_mgr import ToolMgr
from simple_agent.tool.collector import Collector
from simple_agent.db.db import Database
from simple_agent.stream import stream_event


SYSTEM_PROMPT = """You are a planner. Your job is to break down the user's task into sub-tasks
and execute them one at a time.

IMPORTANT: Each sub-task agent can ONLY see the input you give it — it does NOT see
the original user query or previous sub-task results. You MUST include sufficient
context in each sub-task's input so it can work independently. Include:
- The specific goal for this sub-task
- Relevant environment info (working directory, file paths mentioned in results)
- Key findings from previous sub-tasks that this sub-task needs to know
- Any constraints or preferences from the original user request

To define a sub-task: call 'define_task' with the task input (including context).
To finish: call 'determine_state' with state='finished' and a reason.

Example:
1. User asks: "build a test suite for this project"
2. define_task(input="Explore the project at /workspace/project: find all existing test files, identify the test framework used, and check which source modules have no test coverage.")
3. Sub-task executes, results come back showing pytest, 3 modules uncovered
4. define_task(input="Write unit tests for src/utils.py and src/models.py using pytest. These modules were identified as uncovered by the previous exploration. The project uses pytest with fixtures in conftest.py.")
5. Sub-task executes, results come back
6. determine_state(state="finished", reason="Test suite complete for uncovered modules")
"""


class SubTaskProcess:
    agent: Agent
    tools_mgr: ToolMgr
    task_collector: Collector
    state_collector: Collector
    _tools: list
    message: list[AgentMessage]
    _db: Database

    def __init__(self, tools_mgr: ToolMgr | None = None, db: Database | None = None):
        register_custom_models()
        model = get_model("deepseek", "deepseek-v4-pro")
        self.tools_mgr = tools_mgr or ToolMgr()
        self._db = db or Database()
        self.create_task_collector()
        self.create_state_clarify_collector()
        self.wrap_tools()

        agent = Agent(get_api_key=get_api_key)
        agent.set_model(model)
        self._tools: list = []
        self._tools.extend(self.task_collector.tools)
        self._tools.extend(self.state_collector.tools)
        agent.set_tools(self._tools)
        self.agent = agent

    def create_task_collector(self):
        name = "define_task"
        description = "Define a sub-task to be executed. Include all necessary context: goal, environment info, relevant prior findings, and constraints."
        tool_schema = {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The full input for this sub-task, including goal, environment context, relevant prior results, and constraints",
                },
            },
            "required": ["input"],
        }
        self.task_collector = self.tools_mgr.create_collector(
            Task, name, description, tool_schema
        )

    def create_state_clarify_collector(self):
        name = "determine_state"
        description = "Determine the current state. States: finished (task complete), error (task failed)"
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
        self.state_collector = self.tools_mgr.create_collector(
            StateClarification, name, description, tool_schema
        )

    def wrap_tools(self):
        # Wrap define_task to abort agent on call
        task_tool = self.task_collector.tools[0]
        task_original = task_tool.execute
        async def task_execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            res = await task_original(tool_call_id, params, cancel_event, on_update)
            print(f"sub-task defined: {params.get('input', '')[:60]}...")
            self.agent.abort()
            return res
        task_tool.execute = task_execute

        # Wrap determine_state to abort agent on call
        state_tool = self.state_collector.tools[0]
        state_original = state_tool.execute
        async def state_execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            res = await state_original(tool_call_id, params, cancel_event, on_update)
            if self.state_collector.item:
                state = self.state_collector.item[0].state
                print(f"abort on state {state}")
                self.agent.abort()
            return res
        state_tool.execute = state_execute

    def format_result_message(self, task: Task, state: str = "finished") -> list[AgentMessage]:
        from simple_agent.format import format_results
        return format_results(self.tools_mgr, task, status=state)

    async def _step(self, tool_list: list[AgentTool], system_prompt: str, messages: list[AgentMessage], user_prompt: str) -> list[AgentMessage]:
        self.agent.set_system_prompt(system_prompt)
        self.agent.set_tools(tool_list)
        self.agent.replace_messages(messages)
        await self.agent.prompt(user_prompt)
        return self.agent.state.messages

    async def try_sub_task(self) -> Task | StateClarification:
        """Prompt the agent to define a sub-task or complete. Returns sub-Task or None."""
        self.task_collector.clear()
        self.state_collector.clear()

        await self._step(self._tools, SYSTEM_PROMPT, self.message, "now based on the history, determine whether to define a sub task or this task is completed")

        if self.task_collector.item and isinstance(self.task_collector.item[0], Task):
            sub_task = self.task_collector.item[0]
            sub_task.result = []
            return sub_task

        if self.state_collector.item and isinstance(self.state_collector.item[0], StateClarification):
            return self.state_collector[0]

        return None

    async def process(self, task: Task, context: list[AgentMessage] = []) -> list[AgentMessage]:
        self.agent.reset()
        self.agent.subscribe(stream_event)

        index = len(context)
        self.message = context
        self.message.append(UserMessage(content=[TextContent(text=task.input)], timestamp=0))

        if task.result is None:
            task.result = []
        if task.subTasks is None:
            task.subTasks = []

        while True:
            res = await self.try_sub_task()

            if isinstance(res, Task):
                task.subTasks.append(res)
                explore_proc = ExploreProcess(tools_mgr=self.tools_mgr, db=self._db)
                sub_msgs = await explore_proc.process(res)
                self.message.extend(sub_msgs)
                continue

            if isinstance(res, StateClarification):
                collectProc = CollectResultProcess(tools_mgr=self.tools_mgr, db=self._db)
                await collectProc.process(task, self.message[index:])
                break


        self._db.save_task(
            task_type="sub_task",
            task_input=task.input,
            messages=self.message,
            results=task.result,
            status="finished",
        )

        return self.format_result_message(task, state="finished")
