#!/usr/bin/env python3
"""tool-inspect - CLI to inspect tool call results by ID from SQLite database."""

import argparse
import sys
from sqlmodel import Session, create_engine

from simple_agent.tool.db import ToolCallRecord


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
        help="Path to SQLite database file (default: ./tool_log.db)",
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

    if args.list:
        db_path = args.path
        engine = create_engine(f"sqlite:///{db_path}")

        with Session(engine) as session:
            records = session.query(ToolCallRecord).order_by(
                ToolCallRecord.id.desc()
            ).limit(args.limit).all()
            for record in records:
                content_preview = (record.content or "")[:50]
                if len(record.content or "") > 50:
                    content_preview += "..."
                print(f"[{record.id}] {record.tool}: {content_preview}")
        return

    # Require ID for non-list mode
    if args.id is None:
        parser.error("id is required when not using --list")

    db_path = args.path
    engine = create_engine(f"sqlite:///{db_path}")

    # Find and print content for specific ID
    with Session(engine) as session:
        record = session.query(ToolCallRecord).filter(
            ToolCallRecord.id == args.id
        ).first()

        if record:
            sys.stdout.write(record.content or "")
            sys.exit(0)

    # ID not found
    sys.exit(1)


if __name__ == "__main__":
    main()
