#!/usr/bin/env python3
"""tool-inspect - CLI to inspect tool call results by ID."""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Inspect tool call result by ID from tool_log.jsonl"
    )
    parser.add_argument(
        "id",
        type=int,
        help="Tool call ID to inspect",
    )
    parser.add_argument(
        "--path",
        default="./tool_log.jsonl",
        help="Path to tool log file (default: ./tool_log.jsonl)",
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

    if not os.path.exists(args.path):
        print(f"Error: Log file not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    if args.list:
        # List recent tool calls
        lines = []
        with open(args.path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)

        # Show last N entries
        for line in lines[-args.limit:]:
            try:
                entry = json.loads(line)
                content_preview = entry.get("content", "")[:50]
                if len(entry.get("content", "")) > 50:
                    content_preview += "..."
                print(f"[{entry['id']}] {entry['tool']}: {content_preview}")
            except json.JSONDecodeError:
                continue
        return

    # Find and print content for specific ID
    with open(args.path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("id") == args.id:
                    sys.stdout.write(entry.get("content", ""))
                    sys.exit(0)
            except json.JSONDecodeError:
                continue

    # ID not found
    sys.exit(1)


if __name__ == "__main__":
    main()