"""Append-only JSONL logging for agent runtime message changes."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pi.ai.types import AssistantMessage, TextContent, ThinkingContent, ToolCall, ToolResultMessage, UserMessage

if TYPE_CHECKING:
    from pi.agent.types import AgentMessage
    from simple_agent.message_store import MessageEntry


class RuntimeLogger:
    """Write compact, human-readable JSONL records for runtime message changes."""

    def __init__(self, *, log_dir: str | Path = "logs/session_runs"):
        self._log_dir = Path(log_dir)

    def set_log_dir(self, log_dir: str | Path) -> None:
        self._log_dir = Path(log_dir)

    def path(self, session_id: str) -> Path:
        return self._log_dir / f"{session_id}.jsonl"

    def log_handle_running(
        self,
        *,
        session_id: str,
        messages: list["MessageEntry"],
        user_instruction_message: UserMessage,
        assistant_message_id: int,
        assistant_message: AssistantMessage,
        tool_result_entries: list["MessageEntry"],
        next_action: str,
    ) -> None:
        self._write(
            {
                "event": "handle_running",
                "messages": [_message_entry_json(entry) for entry in messages],
                "user_instruction_message": _message_json(user_instruction_message),
                "assistant_message_id": assistant_message_id,
                "assistant_message": _message_json(assistant_message),
                "tool_results": [
                    _tool_result_summary(
                        entry.message,
                        message_id=entry.id,
                    )
                    for entry in tool_result_entries
                    if isinstance(entry.message, ToolResultMessage)
                ],
                "next_action": next_action,
            },
            session_id=session_id,
        )

    def log_handle_compact_result(
        self,
        *,
        session_id: str,
        compact_messages: list["AgentMessage"],
        start_message_id: int,
        end_message_id: int,
        compacted_messages: list["AgentMessage"],
        replacement_messages: list["MessageEntry"],
        next_action: str,
    ) -> None:
        self._write(
            {
                "event": "handle_compact_result",
                "message_scope": {
                    "start_message_id": start_message_id,
                    "end_message_id": end_message_id,
                },
                "compact_messages": [_message_json(message) for message in compact_messages],
                "compacted_messages": [_message_json(message) for message in compacted_messages],
                "replacement_messages": [_message_entry_json(entry) for entry in replacement_messages],
                "next_action": next_action,
            },
            session_id=session_id,
        )

    def _write(self, payload: dict[str, Any], *, session_id: str) -> None:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp_ms": int(time.time() * 1000),
            "session_id": session_id,
            **payload,
        }
        with self.path(session_id).open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


runtime_logger = RuntimeLogger()


def _message_entry_json(entry: "MessageEntry") -> dict[str, Any]:
    return {
        "id": entry.id,
        "message": _message_json(entry.message),
    }


def _message_json(message: "AgentMessage") -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role}
    timestamp = getattr(message, "timestamp", None)
    if timestamp is not None:
        payload["timestamp_ms"] = timestamp

    if isinstance(message, UserMessage):
        payload["content"] = _content_json(message.content)
    elif isinstance(message, AssistantMessage):
        payload["content"] = _content_json(message.content)
        if message.stop_reason is not None:
            payload["stop_reason"] = message.stop_reason
        if message.error_message:
            payload["error_message"] = message.error_message
    elif isinstance(message, ToolResultMessage):
        payload["tool_result"] = _tool_result_summary(message)
    else:
        payload["content"] = _content_json(getattr(message, "content", []))
    return payload


def _content_json(content: object) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return []

    items: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, TextContent):
            items.append({"type": "text", "text": item.text})
        elif isinstance(item, ThinkingContent):
            items.append({"type": "thinking", "thinking": item.thinking})
        elif isinstance(item, ToolCall):
            items.append(
                {
                    "type": "tool_call",
                    "id": item.id,
                    "name": item.name,
                    "arguments": _json_value(item.arguments),
                }
            )
    return items


def _tool_result_summary(
    message: ToolResultMessage,
    *,
    message_id: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "tool_call_id": message.tool_call_id,
        "tool_name": message.tool_name,
    }
    if message_id is not None:
        payload["message_id"] = message_id
    return payload


def _json_value(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value
