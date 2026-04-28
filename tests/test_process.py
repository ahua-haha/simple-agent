"""Tests for the Process abstract base class."""

from __future__ import annotations

import pytest
from dataclasses import dataclass

from simple_agent import Process
from pi.agent import Agent, AgentState
from pi.ai import get_model


@dataclass
class GreetInput:
    name: str


@dataclass
class GreetOutput:
    greeting: str


class GreetingProcess(Process[GreetInput, GreetOutput]):
    """A simple concrete Process implementation for testing."""

    async def process(self, state: AgentState, input: GreetInput) -> tuple[AgentState, GreetOutput]:
        agent = Agent(initial_state=state)
        agent.set_model(get_model("anthropic", "claude-sonnet-4-5"))
        agent.set_system_prompt("You are a greeting assistant. Respond in one sentence.")
        await agent.prompt(f"Say hello to {input.name}.")
        messages = agent.state.messages
        last_text = messages[-1].content[0].text if messages else ""
        return agent.state, GreetOutput(greeting=last_text)


class TestProcess:
    """Tests for Process ABC."""

    def test_process_is_abc(self):
        """Process should be an abstract base class."""
        assert hasattr(Process, "process")

    def test_concrete_process_is_subclass(self):
        """A concrete Process subclass should be a subclass of Process."""
        assert issubclass(GreetingProcess, Process)

    def test_process_is_generic(self):
        """Process should be generic over I and O."""
        assert issubclass(GreetingProcess, Process)
        # Check that the class has the right type parameters
        assert GreetingProcess.__orig_bases__ == (Process[GreetInput, GreetOutput],)

    def test_process_has_abstract_process_method(self):
        """Process.process should be abstract."""
        assert getattr(Process.process, "__isabstractmethod__", False)

    @pytest.mark.asyncio
    async def test_concrete_process_process_returns_tuple(self):
        """A concrete process.process() should return (AgentState, O)."""
        state = AgentState()
        input = GreetInput(name="Alice")

        greeting_process = GreetingProcess()
        result = await greeting_process.process(state, input)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], AgentState)
        assert isinstance(result[1], GreetOutput)

    @pytest.mark.asyncio
    async def test_concrete_process_returns_updated_state(self):
        """The returned state should have additional messages."""
        state = AgentState()
        initial_msg_count = len(state.messages)
        input = GreetInput(name="Bob")

        greeting_process = GreetingProcess()
        returned_state, output = await greeting_process.process(state, input)

        # State should have more messages than before
        assert len(returned_state.messages) > initial_msg_count
        # Returned state should be the same object (mutated in place)
        assert returned_state is state
        # Output should have greeting content
        assert isinstance(output.greeting, str)
