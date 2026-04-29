"""NormalProcess - a Process with all built-in coding tools enabled."""

from __future__ import annotations

from pi.agent import Agent
from pi.ai import get_model
from pi.agent.types import AgentState
from pi.coding.core.tools import create_all_tools

from simple_agent.process import Process
from simple_agent.models import register_custom_models, get_api_key


class NormalProcess(Process[None, None]):
    """A Process with all built-in coding tools (read, bash, edit, write, grep, find, ls).

    This is the standard coding agent process that can:
    - Read and write files
    - Execute shell commands
    - Edit files using diff-based changes
    - Search file contents with grep
    - Find files by name
    - List directory contents

    Input and output are both None - it runs the agent to completion
    and prints the streaming output.
    """

    def __init__(self, cwd: str | None = None):
        """Initialize NormalProcess.

        Args:
            cwd: Working directory for the agent and its tools.
                 Defaults to current working directory if None.
        """
        self.cwd = cwd or "."
        self._tools = create_all_tools(self.cwd)

    def _create_agent(self, state: AgentState | None = None) -> Agent:
        """Create an Agent with all built-in tools configured.

        Args:
            state: Optional AgentState to initialize the agent with.

        Returns:
            Configured Agent instance with tools registered.
        """
        
        register_custom_models()
        model = get_model("deepseek", "deepseek-v4-pro")
        agent = Agent(get_api_key=get_api_key)
        agent.set_tools(list(self._tools.values()))
        agent.set_model(model)
        agent.set_system_prompt("You are a helpful assistant. Respond in one sentence.")
        return agent

    def _on_event(self, event):
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

    async def process(self, state: AgentState, input: None = None) -> tuple[AgentState, None]:
        """Run the agent with the given prompt and stream output.

        Args:
            state: Current AgentState (shared context, may be mutated)
            input: Ignored (None)

        Returns:
            tuple of (updated_agent_state, None)
        """
        register_custom_models()
        agent = self._create_agent(state)
        agent.subscribe(self._on_event)

        try:
            # Get prompt from state messages if exists, otherwise use empty
            await agent.prompt("show me the directory structure")
        except Exception as e:
            print(f"\n[error] {type(e).__name__}: {e}")

        return state, None