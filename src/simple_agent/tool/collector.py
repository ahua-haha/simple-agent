"""Collector - generic tool-based collector for creating and storing type T instances."""

from __future__ import annotations

import asyncio
from typing import Any, Generic, TypeVar, Callable
from dataclasses import dataclass

from pi.ai.types import TextContent
from pi.agent import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pydantic import BaseModel, ValidationError

@dataclass
class CollectorToolResult:
    success: bool
    item: T | None = None
    error: str | None = None

T = TypeVar("T", bound=BaseModel)
def parse_as(data: dict, model_class: type[T]) -> T | None:
    try:
        # Use the passed class to validate the data
        return model_class.model_validate(data)
    except ValidationError as e:
        print(f"Validation failed for {model_class.__name__}: {e}")
        return None

class Collector:
    def __init__(self):
        self.item: list[Any] = []
        self.tools: list[AgentTool] = []

    def add(self, item: Any):
        self.item.append(item)

    def clear(self):
        self.item = []

    def register_record_tool(self, tool : AgentTool):
        self.tools.append(tool)
    
    def create_record_tool(self, model_class: type[T], name: str, description: str, parameters: dict[str, Any], label: str = "") -> AgentTool:
        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            text_str = ""
            try:
                # Use the passed class to validate the data
                item = model_class.model_validate(params)
                self.add(item)
                text_str = f"successsfully execute tool call to record"
            except ValidationError as e:
                text_str = f"Validation failed for {model_class.__name__}: {e}"
            return AgentToolResult(
                content=[TextContent(text=text_str)],
            )

        return AgentTool(
            name=name, description=description, parameters=parameters, label=label, execute=execute
        )