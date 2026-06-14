"""Manual session runner for command-line smoke testing.

Usage:

    uv run python tests/manual_session_cli.py "inspect the project"

The script keeps one temporary session open and reads prompts in a loop.
Use /exit or /quit to stop.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

from pi.agent.types import AgentEvent, AgentMessage
from pi.ai.types import AssistantMessage, TextContent, ThinkingContent, ToolCall, ToolResultMessage, UserMessage

from simple_agent.session import Session


class EventPrinter:
    def __init__(self) -> None:
        self._thinking_pending_word = ""
        self._thinking_open = False

    def print_event(self, event: AgentEvent | dict) -> None:
        if isinstance(event, dict):
            print(f"[event] {event}")
            return

        event_type = getattr(event, "type", None)
        if event_type == "message_update":
            self._print_assistant_message_event(getattr(event, "assistant_message_event", None))
            return

        if event_type == "message_end":
            _print_message(event.message)
            return

        if event_type == "tool_execution_start":
            print(f"[tool call] {event.tool_name}")

    def _print_assistant_message_event(self, assistant_event) -> None:
        event_type = getattr(assistant_event, "type", None)
        if event_type == "thinking_start":
            self._thinking_pending_word = ""
            if not self._thinking_open:
                print("[thinking] ", end="", flush=True)
                self._thinking_open = True
            return

        if event_type == "thinking_delta":
            self._print_thinking_delta(getattr(assistant_event, "delta", ""))
            return

        if event_type == "thinking_end":
            self._flush_thinking_pending_word()
            if self._thinking_open:
                print()
                self._thinking_open = False

    def _print_thinking_delta(self, delta: str) -> None:
        text = self._thinking_pending_word + delta
        if not text:
            return

        words = text.split()
        if text[-1].isspace():
            self._thinking_pending_word = ""
        elif words:
            self._thinking_pending_word = words.pop()
        else:
            self._thinking_pending_word = text

        for word in words:
            print(f"{word} ", end="", flush=True)

    def _flush_thinking_pending_word(self) -> None:
        if self._thinking_pending_word:
            print(f"{self._thinking_pending_word} ", end="", flush=True)
            self._thinking_pending_word = ""


def _print_message(message: AgentMessage) -> None:
    if isinstance(message, UserMessage):
        text = " ".join(item.text for item in message.content if isinstance(item, TextContent))
        print(f"[user] {text}")
        return

    if isinstance(message, ToolResultMessage):
        print(f"[tool result] {message.tool_name}")
        for item in message.content:
            if isinstance(item, TextContent):
                print(item.text)
        return

    if not isinstance(message, AssistantMessage):
        print(f"[message] {type(message).__name__}")
        return

    for item in message.content:
        if isinstance(item, ThinkingContent):
            continue
        elif isinstance(item, ToolCall):
            continue
        elif isinstance(item, TextContent):
            print("[assistant]")
            print(item.text)


async def run_session_from_command_line() -> None:
    initial_input = " ".join(sys.argv[1:]).strip()
    pending_inputs = [initial_input] if initial_input else []

    with tempfile.TemporaryDirectory(prefix="simple-agent-session-") as sessions_dir:
        session = Session(sessions_dir=sessions_dir, workspace_dir=os.getcwd())

        while True:
            if pending_inputs:
                user_input = pending_inputs.pop(0)
                print(f"session input> {user_input}")
            else:
                try:
                    user_input = input("session input> ").strip()
                except EOFError:
                    print()
                    break

            if user_input in {"/exit", "/quit"}:
                break
            if not user_input:
                continue

            await _run_session_turn(session, user_input)


async def _run_session_turn(session: Session, user_input: str) -> None:
    queue = session.run(user_input)
    printer = EventPrinter()

    while True:
        event = await queue.get()
        if event is None:
            break
        printer.print_event(event)


if __name__ == "__main__":
    asyncio.run(run_session_from_command_line())
