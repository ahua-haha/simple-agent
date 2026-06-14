"""Print persisted data for one session.

Usage:

    uv run python tests/manual_session_dump.py session_abc123
    uv run python tests/manual_session_dump.py --db ./sessions/session_abc123.db

The script is read-only. It prints runner metadata and the persisted
message list for the session database.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pi.agent.types import AgentMessage
from pi.ai.types import AssistantMessage, TextContent, ThinkingContent, ToolCall, ToolResultMessage, UserMessage

from simple_agent.db.db import Database


def main() -> None:
    args = _parse_args()
    session_id, db_path = _resolve_session(args)
    db = Database(str(db_path))

    print(f"[session] {session_id}")
    print(f"[db] {db_path}")
    _print_runner_metadata(db, session_id)
    _print_session_record(db, session_id)
    _print_messages(db, session_id)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print persisted data for one simple-agent session")
    parser.add_argument("session_id", nargs="?", help="Session id, for example session_abc123")
    parser.add_argument("--db", help="Path to a session SQLite database")
    parser.add_argument("--base-dir", default="./sessions", help="Directory containing session DB files")
    return parser.parse_args()


def _resolve_session(args: argparse.Namespace) -> tuple[str, Path]:
    if args.db:
        db_path = Path(args.db)
        session_id = args.session_id or db_path.stem
        return session_id, db_path

    if not args.session_id:
        raise SystemExit("session_id is required unless --db is provided")

    return args.session_id, Path(args.base_dir) / f"{args.session_id}.db"


def _print_runner_metadata(db: Database, session_id: str) -> None:
    metadata = db.get_runner_state_metadata(session_id)
    print("[runner metadata]")
    if metadata is None:
        print("  none")
        return

    print(f"  active_user_task_id={metadata.active_user_task_id}")
    print(f"  last_error={metadata.last_error}")
    print(f"  version={metadata.version}")
    print(f"  created_at={metadata.created_at}")
    print(f"  updated_at={metadata.updated_at}")


def _print_session_record(db: Database, session_id: str) -> None:
    record = db.get_session(session_id)
    print("[session metadata]")
    if record is None:
        print("  none")
        return

    for key in ["id", "name", "cursor_id", "created_at", "updated_at"]:
        print(f"  {key}={record.get(key)}")


def _print_messages(db: Database, session_id: str) -> None:
    messages = db.list_runner_messages(session_id)
    print(f"[messages] count={len(messages)}")
    for index, message in enumerate(messages, start=1):
        role = getattr(message, "role", type(message).__name__)
        timestamp = getattr(message, "timestamp", None)
        suffix = f" timestamp={timestamp}" if timestamp is not None else ""
        print(f"{index}. role={role}{suffix}")
        for line in _message_lines(message):
            _print_indented(line)


def _message_lines(message: AgentMessage) -> list[str]:
    if isinstance(message, UserMessage):
        return ["[user]"]

    if isinstance(message, ToolResultMessage):
        return [f"[tool result] {message.tool_name} id={message.tool_call_id}"]

    if isinstance(message, AssistantMessage):
        lines: list[str] = []
        for item in message.content:
            if isinstance(item, TextContent):
                lines.append("[text]")
            elif isinstance(item, ThinkingContent):
                lines.append("[thinking]")
            elif isinstance(item, ToolCall):
                lines.append(f"[tool call] {item.name} id={item.id} args={item.arguments}")
        return lines or ["[assistant]"]

    return [f"[message] {type(message).__name__}"]


def _print_indented(text: str) -> None:
    for line in text.splitlines() or [""]:
        print(f"   {line}")


if __name__ == "__main__":
    main()
