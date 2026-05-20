"""AgentRunState — live state tracked during an agent run, doubles as the cancel_event for agent_loop."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class AgentRunState(asyncio.Event):
    """Mutable state updated by AgentProcess during event stream processing.

    Inherits asyncio.Event so it can be passed directly as agent_loop's cancel_event.
    is_set() combines explicit abort (set()) with should_stop().
    """

    tool_calls: dict[str, list[dict]] = field(default_factory=dict)
    turn_count: int = 0
    finish_reason: str | None = None
    error: str | None = None
    stop_condition: Callable[["AgentRunState"], bool] = field(
        default=lambda s: s.finish_reason is not None
    )

    def __post_init__(self):
        super().__init__()

    def is_set(self) -> bool:
        return super().is_set() or self.should_stop()

    def should_stop(self) -> bool:
        return self.stop_condition(self)
