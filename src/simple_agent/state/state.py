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
    message: list[AgentMessage] = None


class SingleRunTask(BaseModel):
    input: str
    result: list[TextResult] = None
    message: list[AgentMessage] = None
    tasks: list[Task] = None

def generate_state_schema(state_options: dict[str, str]) -> tuple[str, dict[str, Any]]:
    """Generate a tool schema for state clarification.

    Args:
        state_options: dict mapping state names to descriptions.
                       e.g. {"success": "Task completed", "failed": "Task failed"}

    Returns:
        Tuple of (tool_description, parameters_schema).
    """
    state_names = list(state_options.keys())
    state_desc_lines = [f"- {k}: {v}" for k, v in state_options.items()]
    state_description = "Available states:\n" + "\n".join(state_desc_lines)

    description = f"Determine the current state based on context.\n\n{state_description}"
    parameters = {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "description": state_description,
                "enum": state_names,
            },
            "reason": {
                "type": "string",
                "description": "Reason for choosing this state",
            },
        },
        "required": ["state", "reason"],
    }
    return description, parameters

class StateClarification(BaseModel):
    state: str
    reason: str