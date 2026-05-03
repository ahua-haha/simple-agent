"""State module for agent state management."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel

from pi.agent.types import AgentMessage, AgentToolResult
from pi.ai.types import ToolCall


class ToolExecMessage(BaseModel):
    input: ToolCall
    output: str


class TextResult(BaseModel):
    desc: str
    toolCallLogID: list[int]


TEXT_RESULT_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "desc": {"type": "string", "description": "Description of the result"},
        "toolCallLogID": {"type": "array", "items": {"type": "integer"}, "description": "List of tool call log IDs"},
    },
    "required": ["desc", "toolCallLogID"],
}


class Task(BaseModel):
    input: str
    result: list[TextResult]


class SingleRunTask(BaseModel):
    input: str
    result: list[TextResult] = None
    message: list[AgentMessage] = None
    tasks: list[Task] = None