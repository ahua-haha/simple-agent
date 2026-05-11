"""Shared event streaming — prints agent events to stdout."""

from pi.agent import (
    AgentEndEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from pi.agent.types import AssistantMessageEvent


def stream_event(event) -> None:
    """Print agent events in streaming mode. Subscribe via agent.subscribe(stream_event)."""
    if isinstance(event, MessageUpdateEvent):
        ae = event.assistant_message_event
        if isinstance(ae, AssistantMessageEvent):
            if ae.type == "thinking_start":
                print("<thinking>", end="\n", flush=True)
            elif ae.type == "text_start":
                print("<resp>", end="\n", flush=True)
            elif ae.type == "thinking_end":
                print("\n</thinking>", end="\n", flush=True)
            elif ae.type == "text_end":
                print("\n</resp>", end="\n", flush=True)
            elif ae.type == "text_delta":
                print(ae.delta, end="", flush=True)
            elif ae.type == "thinking_delta":
                print(ae.delta, end="", flush=True)
    elif isinstance(event, ToolExecutionStartEvent):
        print(f"\n[tool: {event.tool_name}({event.args})]", flush=True)
    elif isinstance(event, ToolExecutionEndEvent):
        pass
    elif isinstance(event, AgentEndEvent):
        print("\n[agent done]", flush=True)
