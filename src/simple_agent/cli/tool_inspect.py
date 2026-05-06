#!/usr/bin/env python3
"""tool-inspect - CLI to inspect tool call results by ID from SQLite database."""

import argparse
import sys

from simple_agent.db.db import Database


def main():
    parser = argparse.ArgumentParser(
        description="Inspect tool call result by ID from SQLite database"
    )
    parser.add_argument(
        "id",
        type=int,
        nargs="?",
        help="Tool call ID to inspect (required if not using --list)",
    )
    parser.add_argument(
        "--path",
        default="./data/tool_log.db",
        help="Path to SQLite database file (default: ./data/tool_log.db)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List recent tool calls with IDs and first 50 chars",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit for --list (default: 10)",
    )

    args = parser.parse_args()

    db = Database(args.path)

    if args.list:
        records = db.list_tool_calls(args.limit)
        for record in records:
            content_preview = (record.content or "")[:50]
            if len(record.content or "") > 50:
                content_preview += "..."
            print(f"[{record.id}] {record.tool}: {content_preview}")
        return

    # Require ID for non-list mode
    if args.id is None:
        parser.error("id is required when not using --list")

    # Find and print content for specific ID
    record = db.get_tool_call(args.id)

    if record:
        sys.stdout.write(record.raw_output or "")
        sys.exit(0)

    # ID not found
    sys.exit(1)


if __name__ == "__main__":
    main()
