#!/usr/bin/env python3
"""task-msg-log - CLI to display task message history in readable format."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from typing import Any

from pi.agent.types import AgentMessage
from pi.ai import AssistantMessage, ToolResultMessage, UserMessage

from simple_agent.state.state import TextResult


DIVIDER = "─" * 55
HEADER_DIVIDER = "═" * 55


def format_timestamp(ts: int) -> str:
    """Format Unix timestamp to readable string."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def truncate(s: str, limit: int) -> str:
    """Truncate string to limit, appending length indicator."""
    if len(s) <= limit:
        return s
    return s[:limit] + f"... (len={len(s)})"

def list_tasks(db, limit: int = 10):
    """List recent tasks."""
    tasks = db.list_tasks(limit=limit)

    if not tasks:
        print("No tasks found.")
        return

    for task in tasks:
        task_id = task["id"]
        task_type = task.get("type", "unknown")
        task_status = task.get("status", "unknown")
        task_input = (task.get("input") or "")[:50]
        created_at = format_timestamp(task.get("created_at", 0))
        if len(task.get("input") or "") > 50:
            task_input += "..."

        print(f"[{task_id}] {task_type} | {task_status} | {created_at}")
        print(f"    {task_input}")
        print()


def display_task(db, task_id: int, truncate_limit: int = 200):
    """Display full message flow for a task."""
    task = db.get_task(task_id)

    if not task:
        print(f"Task #{task_id} not found.")
        sys.exit(1)

    # Header
    print(HEADER_DIVIDER)
    task_type = task.get("type", "unknown")
    task_status = task.get("status", "unknown")
    created_at = format_timestamp(task.get("created_at", 0))
    print(f"Task #{task_id} | {task_type} | {task_status} | {created_at}")
    print(HEADER_DIVIDER)
    print()

    # Get messages from task - already deserialized dict from db.get_task()
    messages: list[AgentMessage] = task.get("messages", [])
    if not messages:
        print("(No messages)")
    else:
        for msg in messages:
            if isinstance(msg, UserMessage):
                text = msg.content[0].text if msg.content else ""
                print(f"[USER]")
                print(text)
                print()
                print(DIVIDER)
                print()

            elif isinstance(msg, AssistantMessage):
                from pi.ai.types import TextContent, ThinkingContent, ToolCall
                parts = []
                for item in msg.content:
                    if isinstance(item, TextContent):
                        parts.append(item.text)
                    elif isinstance(item, ThinkingContent):
                        parts.append(f"<thinking>\n{item.thinking}\n</thinking>")
                    elif isinstance(item, ToolCall):
                        args_str = json.dumps(item.arguments)
                        args_str = truncate(args_str, truncate_limit)
                        parts.append(f"[TOOL CALL] {item.name}\n  args: {args_str}")
                if parts:
                    print(f"[ASSISTANT]")
                    print("\n".join(parts))
                    print()
                    print(DIVIDER)
                    print()

            elif isinstance(msg, ToolResultMessage):
                from pi.ai import ToolResultMessage as TRMsg
                print(f"[TOOL CALL] {msg.tool_name}")
                print("  ↓")
                print("[TOOL RESULT]")
                result_text = msg.content[0].text if msg.content else ""
                print(truncate(result_text, truncate_limit))
                print()
                print(DIVIDER)
                print()

    # Print task results if any
    results:list[TextResult] = task.get("results", [])
    if results:
        print(HEADER_DIVIDER)
        print("[TASK RESULTS]")
        for i, result in enumerate(results):
            desc = result.desc
            tool_call_ids = result.toolCallLogID
            print(f"  Result {i + 1}: {truncate(desc, truncate_limit)}")
            if tool_call_ids:
                print(f"    toolCallLogID: {tool_call_ids}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Display task message history in readable format"
    )
    parser.add_argument(
        "task_id",
        type=int,
        nargs="?",
        help="Task ID to inspect (required if not using --list)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List recent tasks instead of inspecting a specific task",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit number of tasks shown (default: 10, only with --list)",
    )
    parser.add_argument(
        "--truncate",
        type=int,
        default=200,
        help="Truncation limit for tool results (default: 200 characters)",
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to SQLite database file",
    )

    args = parser.parse_args()

    # Import Database directly - lazy TOOL_MGR in globals.py prevents circular init
    from simple_agent.db.db import Database
    db = Database(args.path)

    if args.list:
        list_tasks(db, args.limit)
    elif args.task_id is not None:
        display_task(db, args.task_id, args.truncate)
    else:
        parser.error("task_id is required when not using --list")


if __name__ == "__main__":
    main()
