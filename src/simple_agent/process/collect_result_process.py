"""CollectResultProcess - synthesize TextResults from exploration history."""

from __future__ import annotations

from pi.agent import Agent, AgentTool
from pi.ai import get_model
from pi.agent.types import AgentMessage

from simple_agent.db.db import Database
from simple_agent.stream import stream_event
from simple_agent.models import register_custom_models, get_api_key
from simple_agent.state.state import TEXT_RESULT_JSON_SCHEMA, Task, SingleRunTask, TextResult
from simple_agent.tool.tool_mgr import ToolMgr
from simple_agent.tool.collector import Collector


SYSTEM_PROMPT = """You are a result synthesizer. Review the conversation history,
use bash commands like tool-inspect, grep, sed, head to inspect
tool results, and record each useful outcome as TextResult.
Focus on WHAT was accomplished, not HOW.
When done, respond with only FINISH. Do NOT generate verbose output.

Examples:
- bash("tool-inspect 3 | grep 'function'") to filter tool call 3 result
- bash("tool-inspect 5 | head -10") to get first 10 lines
- record_textresult(desc="Found main.py", toolCallLogID=[3]) to record a result
"""


class CollectResultProcess:
    agent: Agent
    collector: Collector
    tools: list[AgentTool]
    tools_mgr: ToolMgr
    message: list[AgentMessage]
    _db: Database


    def __init__(self, tools_mgr: ToolMgr | None = None, db: Database | None = None):
        register_custom_models()
        # model = get_model("minimax-cn", "MiniMax-M2.7")
        model = get_model("deepseek", "deepseek-v4-pro")
        self.tools_mgr = tools_mgr or ToolMgr()
        self._db = db or Database()
        self.collector = self.tools_mgr.create_collector(
            model_class=TextResult,
            name=f"record_textresult",
            description="Record a TextResult instance with the tool call log ID referencing related tool executions",
            parameters=TEXT_RESULT_JSON_SCHEMA,
        )
        self.message = []

        agent = Agent(get_api_key=get_api_key)
        agent.set_model(model)

        # Get bash tools from ToolMgr (for tool-inspect)
        bash_tools = self.tools_mgr.create_all_tools(".")
        # Get record_textresult tool from collector
        all_tools = bash_tools
        all_tools.extend(self.collector.tools)
        self.tools = all_tools

        self.agent = agent

    async def _step(self, system_prompt: str, tool_list: list, user_prompt: str):
        self.agent.set_system_prompt(system_prompt)
        self.agent.set_tools(tool_list)
        self.agent.replace_messages(self.message)
        await self.agent.prompt(user_prompt)
        self.message = self.agent.state.messages

    async def process(self, task: Task, context: list[AgentMessage]) -> list[AgentMessage]:
        """Synthesize TextResults from task's message history.

        Args:
            task: Task or SingleRunTask with message history.
                  After processing, task.result will contain collected TextResults.
        """
        self.agent.reset()
        self.agent.subscribe(stream_event)

        index = len(context)
        self.message = context
        await self._step(SYSTEM_PROMPT, self.tools, "Please review the conversation history and record all useful results as TextResult using the record_textresult tool. When done, respond with only FINISH.")

        if self.collector.item:
            task.result = list(self.collector.item)

        self._db.save_task(
            task_type="collect result",
            task_input=task.input,
            messages=self.message,
            results=task.result,
            status="finished",
        )

        return self.message[index:]