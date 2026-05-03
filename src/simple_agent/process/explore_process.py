
from pi.agent import Agent
from pi.ai import get_model
from pi.agent.types import AgentMessage, AgentState
from pi.coding.core.tools import create_all_tools

from simple_agent.process.process import Process
from simple_agent.models import register_custom_models, get_api_key
from simple_agent.state.state import Task
from simple_agent.tool.tool_mgr import ToolMgr
import time


class ExploreProcess:
    agent: Agent


    def __init__(self):
        register_custom_models()
        model = get_model("deepseek", "deepseek-v4-pro")
        tools_mgr = ToolMgr()

        agent = Agent(get_api_key=get_api_key)
        agent.set_model(model)
        agent.set_tools(tools_mgr.create_all_tools("."))
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

    async def process(self, task: Task):
        self.agent.reset()
        self.agent.replace_messages(task.message)
        self.agent.subscribe(self.on_event)
        await self.agent.prompt(task.input)
        return