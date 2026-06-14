#!/usr/bin/env python3
"""tool-inspect - CLI to inspect runner tool call results by ID."""

import argparse
import sys

from sqlmodel import Session, select

from simple_agent.db.db import Database
from simple_agent.state.state import RunnerToolCallRecord


def main():
    parser = argparse.ArgumentParser(
        description="Inspect tool call result by ID from SQLite database"
    )
    parser.add_argument(
        "id",
        type=int,
        nargs="?",
        help="Runner tool call log ID to inspect (required if not using --list)",
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to SQLite database file",
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
        with Session(db._engine) as session:
            records = session.exec(
                select(RunnerToolCallRecord)
                .order_by(RunnerToolCallRecord.id.desc())
                .limit(args.limit)
            ).all()
        for record in records:
            content = record.tool_result_json or ""
            content_preview = content[:50]
            if len(content) > 50:
                content_preview += "..."
            print(f"[{record.id}] {record.tool_name}: {content_preview}")
        return

    # Require ID for non-list mode
    if args.id is None:
        parser.error("id is required when not using --list")

    # Find and print content for specific ID
    with Session(db._engine) as session:
        record = session.get(RunnerToolCallRecord, args.id)

    if record:
        sys.stdout.write(record.tool_result_json or "")
        sys.exit(0)

    # ID not found
    sys.exit(1)


if __name__ == "__main__":
    main()
