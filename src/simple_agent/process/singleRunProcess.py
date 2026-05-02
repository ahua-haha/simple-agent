
from pi.agent import Agent
from pi.ai import get_model
from pi.agent.types import AgentMessage, AgentState
from pi.coding.core.tools import create_all_tools

from simple_agent.process.process import Process
from simple_agent.models import register_custom_models, get_api_key
from simple_agent.state.state import SingleRunTask


def on_event(event):
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
        print(f"\n[tool end: {event.tool_name} → result={event.result}]", flush=True)
    elif event.type == "agent_end":
        print("\n[agent done]", flush=True)

class SingleRunProcess:
    agent: Agent


    def __init__(self):
        register_custom_models()
        model = get_model("deepseek", "deepseek-v4-pro")

        agent = Agent(get_api_key=get_api_key)
        agent.set_model(model)
        self.agent = agent

    async def process(self, task: SingleRunTask):
        self.agent.reset()
        self.agent.replace_messages(task.message)
        self.agent.subscribe(on_event)
        await self.agent.prompt(task.input)
        return