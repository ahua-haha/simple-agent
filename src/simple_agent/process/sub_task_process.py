"""SubTaskProcess - planner agent that decomposes tasks into sub-tasks."""

from __future__ import annotations

import asyncio
from typing import Any

from pi.agent import Agent, AgentToolResult, AgentToolUpdateCallback
from pi.ai import UserMessage, TextContent, get_model
from pi.ai.types import AssistantMessage, ToolResultMessage
from pi.agent.types import AgentMessage

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

When to define a sub-task:
- The task requires multiple exploration, retrieval, or research steps
- A clear, scoped piece of work can be extracted
- More investigation is needed before giving a final answer

To define a sub-task: call 'define_task' with the task input.
To finish: call 'determine_state' with state='finished' and a reason.

Always review the results of completed sub-tasks before defining the next one.
When sufficient results are gathered, call determine_state to finish.

Example flow:
1. User asks: "build a test suite for this project"
2. You call: define_task(input="explore existing tests and project structure")
3. Sub-task executes, results come back
4. You call: define_task(input="write unit tests for the uncovered modules")
5. Sub-task executes, results come back
6. You call: determine_state(state="finished", reason="test suite complete")
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
        description = "Define a sub-task to be executed"
        tool_schema = {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The input description for this sub-task",
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

    def prune_define_task(self):
        last_two = self.message[-2:]
        if isinstance(last_two[0], AssistantMessage) and isinstance(last_two[1], ToolResultMessage) and last_two[1].tool_name == "define_task":
            print("prune define_task tool call")
            del self.message[-2:]

    def prune_determine_state(self):
        last_two = self.message[-2:]
        if isinstance(last_two[0], AssistantMessage) and isinstance(last_two[1], ToolResultMessage) and last_two[1].tool_name == "determine_state":
            print("prune determine_state tool call")
            del self.message[-2:]

    def format_result_message(self, task: Task) -> list[AgentMessage]:
        result = [UserMessage(content=[TextContent(text=task.input)], timestamp=0)]
        tool_log_id: list[int] = []
        for res in task.result or []:
            tool_log_id.extend(res.toolCallLogID)
        result.extend(self.tools_mgr.get_all_messages(tool_log_id))
        return result

    async def _step(self, system_prompt: str, tool_list: list):
        self.agent.set_system_prompt(system_prompt)
        self.agent.set_tools(tool_list)
        self.agent.replace_messages(self.message)
        await self.agent.continue_()
        self.message = self.agent.state.messages

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
            self.task_collector.clear()
            self.state_collector.clear()

            await self._step(SYSTEM_PROMPT, self._tools)

            # Check for sub-task definition
            if self.task_collector.item:
                self.prune_define_task()

                sub_task = self.task_collector.item[-1]
                sub_task.result = []
                task.subTasks.append(sub_task)

                explore_proc = ExploreProcess(tools_mgr=self.tools_mgr, db=self._db)
                sub_msgs = await explore_proc.process(sub_task, self.message)
                self.message.extend(sub_msgs)
                continue

            # Check for completion
            if self.state_collector.item:
                self.prune_determine_state()
                break

            # No tool called — agent gave plain text response, treat as finish
            break

        collectProc = CollectResultProcess(tools_mgr=self.tools_mgr, db=self._db)
        await collectProc.process(task, self.message[index:])

        self._db.save_task(
            task_type="sub_task",
            task_input=task.input,
            messages=self.message,
            results=task.result,
            status="finished",
        )

        return self.format_result_message(task)
