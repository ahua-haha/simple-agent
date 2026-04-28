"""Process abstract base class for typed agent processing units."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pi.agent import Agent
from pi.agent.types import AgentState

I = TypeVar("I")
O = TypeVar("O")


class Process(ABC, Generic[I, O]):
    """Abstract base class for one unit of agent processing work.

    Each Process is a self-contained unit that:
    - Receives typed input I and current AgentState
    - Creates and configures an Agent internally
    - Runs the agent to process the input
    - Returns (updated_state, typed_output O)

    Subclasses fully own their agent configuration — model, tools,
    system_prompt, and output parsing are all implementation details.
    """

    @abstractmethod
    async def process(self, state: AgentState, input: I) -> tuple[AgentState, O]:
        """Process the input and return updated state and output.

        Args:
            state: Current AgentState (shared context, may be mutated)
            input: Typed input specific to this Process implementation

        Returns:
            tuple of (updated_agent_state, output of type O)
        """
        ...
