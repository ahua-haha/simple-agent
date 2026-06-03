"""Manual session runner that pauses after every turn and dumps DB messages.

Usage:

    uv run python tests/manual_pause_turn_cli.py "inspect the project"

Commands after a pause:

    /continue    Continue the paused task with session.run(None)
    /exit        Stop the script
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from typing import Any

from pi.agent.types import AgentEvent, AgentMessage
from pi.ai.types import AssistantMessage, TextContent, ThinkingContent, ToolCall, ToolResultMessage, UserMessage

from simple_agent.session import Session


async def run_session_from_command_line() -> None:
    initial_input = " ".join(sys.argv[1:]).strip()
    pending_inputs: list[str | None] = [initial_input] if initial_input else []

    with tempfile.TemporaryDirectory(prefix="simple-agent-pause-turn-") as sessions_dir:
        session = Session(base_dir=sessions_dir)
        print(f"[session] {session.id}")

        while True:
            if pending_inputs:
                user_input = pending_inputs.pop(0)
                print(f"session input> {user_input}")
            else:
                try:
                    raw_input = input("session input> ").strip()
                except EOFError:
                    print()
                    break

                if raw_input in {"/exit", "/quit"}:
                    break
                if raw_input == "/continue":
                    user_input = None
                elif raw_input:
                    user_input = raw_input
                else:
                    continue

            await _run_until_paused(session, user_input)
            _print_database_messages(session)


async def _run_until_paused(session: Session, user_input: str | None) -> None:
    def pause_on_turn_end(event: AgentEvent) -> None:
        if getattr(event, "type", None) == "turn_end":
            session.pause()

    session._runner.add_hook("turn_end", pause_on_turn_end)
    queue = session.run(user_input)

    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            _print_event(event)
    finally:
        session._runner.remove_hook("turn_end", pause_on_turn_end)


def _print_event(event: AgentEvent | dict) -> None:
    if isinstance(event, dict):
        print(f"[event] {event}")
        return

    event_type = getattr(event, "type", None)
    if event_type == "tool_execution_start":
        print(f"[tool call] {event.tool_name}")
        return
    if event_type == "message_end":
        _print_message(event.message)
        return
    if event_type == "turn_end":
        print("[event] turn_end, paused")
        return
    if event_type == "agent_end":
        print("[event] agent_end")


def _print_database_messages(session: Session) -> None:
    entries = session._db.list_runner_message_entries(session.id)
    print(f"[db messages] count={len(entries)}")
    for index, entry in enumerate(entries, start=1):
        role = getattr(entry.message, "role", type(entry.message).__name__)
        print(f"{index}. seq={entry.seq} role={role}")
        for line in _message_lines(entry.message):
            _print_indented(line)


def _print_indented(text: str) -> None:
    for line in text.splitlines() or [""]:
        print(f"   {line}")


def _print_message(message: AgentMessage) -> None:
    for line in _message_lines(message):
        print(line)


def _message_lines(message: Any) -> list[str]:
    if isinstance(message, UserMessage):
        text = _text_content(message.content)
        return [f"[user] {text}"]

    if isinstance(message, ToolResultMessage):
        return [f"[tool result] {message.tool_name} id={message.tool_call_id}"]

    if isinstance(message, AssistantMessage):
        lines: list[str] = []
        for item in message.content:
            if isinstance(item, TextContent):
                lines.append(f"[assistant] {item.text}")
            elif isinstance(item, ThinkingContent):
                lines.append(f"[thinking] {item.thinking}")
            elif isinstance(item, ToolCall):
                lines.append(f"[tool call] {item.name} id={item.id} args={item.arguments}")
        return lines or ["[assistant]"]

    return [f"[message] {type(message).__name__}"]


def _text_content(content: list[Any]) -> str:
    return "\n".join(item.text for item in content if isinstance(item, TextContent))


if __name__ == "__main__":
    asyncio.run(run_session_from_command_line())
