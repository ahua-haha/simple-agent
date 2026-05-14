
import asyncio
from typing import Any

from pi.agent import Agent, AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi.ai import ToolCall, get_model
from pi.ai.types import AssistantMessage, TextContent, ToolResultMessage, UserMessage
from pi.agent.types import AgentMessage, AgentState
from pi.coding.core.tools import create_all_tools

from simple_agent.db.db import Database
from simple_agent.tool.tool_mgr import ToolMgr
from simple_agent.process.process import Process
from simple_agent.models import register_custom_models, get_api_key
from simple_agent.state.state import TEXT_RESULT_JSON_SCHEMA, Task, TextResult, StateClarification
from simple_agent.tool.tool_mgr import ToolMgr
from simple_agent.tool.collector import Collector
from simple_agent.process.collect_result_process import CollectResultProcess
from simple_agent.stream import stream_event
import time
from pprint import pprint


SYSTEM_PROMPT = """You are a helpful assistant. your job is to use the avaliable tools to explore and retrieval the infomation.
<important>
When the task is complete and no further tool calls are required, you MUST use 'determine_state' tool to determine the state BEFORE your final response.
</important>

<example>
tool call 1 ...
tool call 1 result ...
tool call 2 ...
tool call 2 result ...
tool call 3 ...
tool call 3 result ...

Now the context infomation is complete. use 'determine_state' tool call to determine the state
Final response: ...
</example>

"""

class ExploreProcess:
    agent: Agent
    tools_mgr: ToolMgr
    state_collector: Collector
    _db: Database


    def __init__(self, tools_mgr: ToolMgr | None = None, db: Database | None = None):
        register_custom_models()
        # model = get_model("minimax-cn", "MiniMax-M2.7")
        model = get_model("deepseek", "deepseek-v4-pro")
        self.tools_mgr = tools_mgr or ToolMgr()
        self._db = db or Database()
        self.create_state_clarify_collector()
        self.wrap_tools()

        agent = Agent(get_api_key=get_api_key)
        agent.set_model(model)
        all_tools = self.tools_mgr.create_all_tools(".")
        all_tools.extend(self.state_collector.tools)
        agent.set_tools(all_tools)
        agent.set_system_prompt(SYSTEM_PROMPT)
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
    
    def format_task_message(self, task: Task) -> list[AgentMessage]:
        tool_log_id = []
        for res in task.result:
            tool_log_id.extend(res.toolCallLogID)
        pprint(tool_log_id)

        messages = [UserMessage(content=[TextContent(text=task.input)], timestamp=0)]
        messages.extend(self.tools_mgr.get_all_messages(tool_log_id))
        return messages

    async def _step(self, task: Task):
        self.agent.replace_messages(self.message)
        await self.agent.prompt(task.input)
        self.message = self.agent.state.messages
        self.prune_message()

    async def process(self, task: Task, context: list[AgentMessage]) -> list[AgentMessage]:
        self.agent.reset()
        # self.agent.replace_messages(task.message)
        self.agent.subscribe(stream_event)
        index = len(context)
        self.message = context

        await self._step(task)

        collectProc = CollectResultProcess(tools_mgr=self.tools_mgr, db=self._db)
        await collectProc.process(task, self.message[index:])

        # Save task to history
        self._db.save_task(
            task_type="explore",
            task_input=task.input,
            messages=self.message,
            results=task.result,
            status="finished",
        )

        return self.format_task_message(task)