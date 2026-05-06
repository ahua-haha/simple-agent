"""CollectResultProcess - synthesize TextResults from exploration history."""

from __future__ import annotations

from pi.agent import Agent
from pi.ai import get_model
from pi.agent.types import AgentMessage

from simple_agent.db.db import Database
from simple_agent.globals import TOOL_MGR
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
    tools_mgr: ToolMgr
    message: list[AgentMessage]
    _db: Database


    def __init__(self):
        register_custom_models()
        # model = get_model("minimax-cn", "MiniMax-M2.7")
        model = get_model("deepseek", "deepseek-v4-pro")
        self.tools_mgr = TOOL_MGR
        self._db = Database()
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

        agent.set_tools(all_tools)
        agent.set_system_prompt(SYSTEM_PROMPT)
        self.agent = agent

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
    
    async def _step(self, task: Task):
        self.agent.replace_messages(self.message)
        await self.agent.prompt("Please review the conversation history and record all useful results as TextResult using the record_textresult tool. When done, respond with only FINISH.")
        self.message = self.agent.state.messages

    async def process(self, task: Task, context: list[AgentMessage]) -> list[AgentMessage]:
        """Synthesize TextResults from task's message history.

        Args:
            task: Task or SingleRunTask with message history.
                  After processing, task.result will contain collected TextResults.
        """
        self.agent.reset()
        self.agent.subscribe(self.on_event)

        index = len(context)
        self.message = context
        await self._step(task)

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