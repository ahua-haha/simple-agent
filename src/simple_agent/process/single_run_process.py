
import asyncio
from typing import Any

from pi.agent import Agent, AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi.ai import get_model
from pi.agent.types import AgentMessage, AgentState
from pi.coding.core.tools import create_all_tools

from simple_agent.process.process import Process
from simple_agent.process.explore_process import ExploreProcess
from simple_agent.models import register_custom_models, get_api_key
from simple_agent.state.state import SingleRunTask, Task, TextResult, StateClarification
from simple_agent.tool.tool_mgr import ToolMgr
from simple_agent.tool.collector import Collector
from simple_agent.globals import TOOL_MGR
import time


SYSTEM_PROMPT = """You are a planner assistant. Your job is to review task context and message history,
then decide whether to define a sub-task or give a final response.

When to define a sub-task:
- Task requires multi-step exploration or research
- Intermediate results need to be captured
- Clear boundary between subtasks exists

When to give final response (NO tool calls):
- Task is complete with sufficient results
- No further tool calls needed
- Ready to summarize findings

To define a sub-task: call 'define_task' tool with input and scope_index.
To finish: simply provide your final response text (no tool calls).
"""


class SingleRunProcess:
    agent: Agent
    tools_mgr: ToolMgr
    task_collector: Collector
    _sub_task_defined: bool


    def __init__(self):
        register_custom_models()
        model = get_model("deepseek", "deepseek-v4-pro")
        self.tools_mgr = TOOL_MGR

        self.create_task_collector()
        self.wrap_tools()

        agent = Agent(get_api_key=get_api_key)
        agent.set_model(model)
        agent.set_tools(self.task_collector.tools)
        agent.set_system_prompt(SYSTEM_PROMPT)
        self.agent = agent

    def create_task_collector(self):
        name = "define_task"
        description = "Define a sub-task to be executed. Creates a Task instance with message history from parent."
        tool_schema = {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The input description for this sub-task",
                },
                "scope_index": {
                    "type": "integer",
                    "description": "The message index where this task's scope begins (0 = from start)",
                },
            },
            "required": ["input", "scope_index"],
        }
        self.task_collector = self.tools_mgr.create_collector(
            Task, name, description, tool_schema
        )

    def wrap_tools(self):
        # Wrap define_task to abort agent when called
        task_tool = self.task_collector.tools[0]
        self._sub_task_defined = False
        original_task_execute = task_tool.execute
        async def task_execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            res = await original_task_execute(tool_call_id, params, cancel_event, on_update)
            self._sub_task_defined = True
            print(f"sub-task defined: {params.get('input', '')[:50]}...")
            self.agent.abort()
            return res
        task_tool.execute = task_execute


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
            print(f"{event.result.content[0].text}")
        elif event.type == "agent_end":
            print("\n[agent done]", flush=True)

    async def process(self, task: SingleRunTask):
        self.agent.reset()
        self.agent.replace_messages(task.message or [])
        self.agent.subscribe(self.on_event)

        # Initialize task result and tasks list
        if task.result is None:
            task.result = []
        if task.tasks is None:
            task.tasks = []

        # Main loop: keep running until no more sub-tasks defined
        while True:
            # Clear collectors at start of each iteration
            self.task_collector.clear()
            self._sub_task_defined = False

            # Run agent - will abort if define_task is called
            await self.agent.prompt(task.input)
            task.message = self.agent.state.messages

            # Check if sub-task was defined
            if self._sub_task_defined and self.task_collector.item:
                # Create sub-task from collector
                child_task = self.task_collector.item[-1]
                # Copy parent messages to child (current messages at time of define_task call)
                child_task.message = task.message
                child_task.result = []
                task.tasks.append(child_task)

                # Run child task via ExploreProcess
                explore_proc = ExploreProcess()
                await explore_proc.process(child_task)

                # Merge child's result to parent
                if child_task.result:
                    task.result.extend(child_task.result)

                # Continue loop for next decision
                continue

            # No sub-task defined - agent gave final response, we're done
            break

        return