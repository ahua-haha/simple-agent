
from pi.agent import Agent
from pi.ai import get_model
from pi.agent.types import AgentMessage, AgentState
from pi.coding.core.tools import create_all_tools

from simple_agent.process.process import Process
from simple_agent.models import register_custom_models, get_api_key
from simple_agent.state.state import Task, TextResult
from simple_agent.tool.tool_mgr import ToolMgr
from simple_agent.tool.collector import Collector
from simple_agent.process.collect_result_process import CollectResultProcess
import time


SYSTEM_PROMPT = """You are a helpful assistant. your job is to use the avaliable tools to explore and retrieval the infomation.

IMPORTANT:When done, respond with only "FINISH". Do NOT generate verbose output.

"""

class ExploreProcess:
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
        all_tools = self.tools_mgr.create_all_tools(".")
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

    async def process(self, task: Task):
        self.agent.reset()
        # self.agent.replace_messages(task.message)
        self.agent.subscribe(self.on_event)
        await self.agent.prompt(task.input)
        self.tools_mgr.flush()
        task.message = self.agent.state.messages

        collectProc = CollectResultProcess()
        await collectProc.process(task)
        print(task.result)

        return