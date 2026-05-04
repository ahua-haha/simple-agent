"""CollectResultProcess - synthesize TextResults from exploration history."""

from __future__ import annotations

from pi.agent import Agent
from pi.ai import get_model
from pi.agent.types import AgentMessage

from simple_agent.models import register_custom_models, get_api_key
from simple_agent.state.state import Task, SingleRunTask, TextResult
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


    def __init__(self):
        register_custom_models()
        model = get_model("deepseek", "deepseek-v4-pro")
        self.tools_mgr = ToolMgr()
        self.collector = self.tools_mgr.create_collector()

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
            print(f"{event.result.content[0].text}")
        elif event.type == "agent_end":
            print("\n[agent done]", flush=True)

    async def process(self, task: Task | SingleRunTask):
        """Synthesize TextResults from task's message history.

        Args:
            task: Task or SingleRunTask with message history.
                  After processing, task.result will contain collected TextResults.
        """
        self.agent.reset()
        self.agent.replace_messages(task.message)
        self.agent.subscribe(self.on_event)

        # Prompt to trigger the result collection
        await self.agent.prompt("Please review the conversation history and record all useful results as TextResult using the record_textresult tool. When done, respond with only FINISH.")

        # Wait for FINISH or agent to end
        # The _finish_detected flag is set by on_event handler

        # Transfer collected items to task.result
        if self.collector.item:
            task.result = list(self.collector.item)

        return task