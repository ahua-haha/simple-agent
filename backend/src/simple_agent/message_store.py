"""Shared message entry model."""

from __future__ import annotations

from dataclasses import dataclass

from pi.agent.types import AgentMessage


@dataclass(frozen=True)
class MessageEntry:
    id: int
    message: AgentMessage
