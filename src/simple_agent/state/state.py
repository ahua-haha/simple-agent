"""State module for agent state management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pi.agent.types import AgentMessage
from pi.ai.types import AssistantMessage, ToolResultMessage, ToolCall


@dataclass
class ToolExecMessage:
    input: ToolCall
    output: ToolResultMessage

@dataclass
class Task:
    input: str
    result: list[TextResult]


# @dataclass
# class FileContentResult:
#     desc:str
#     filePath: str
#     lineRanges: list[tuple[int, int]]
#     toolCallLogID: list[int]

# @dataclass
# class ModifyResult:
#     desc:str
#     files: list[str]
#     toolCallLogID: list[int]

@dataclass
class TextResult:
    desc:str
    toolCallLogID: list[int]

class SingleRunTask:
    input: str
    result: list[TextResult]

    message: list[AgentMessage]
    tasks: list[Task]
