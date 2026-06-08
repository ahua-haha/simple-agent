"""repo-index CLI for inspecting AgentIndex data."""

from __future__ import annotations

import argparse
import sys

from simple_agent.index import AgentIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect repository AgentIndex data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tree_parser = subparsers.add_parser("tree", help="Render the AgentIndex tree")
    tree_parser.add_argument(
        "--db",
        required=True,
        help="Path to the AgentIndex SQLite database",
    )
    tree_parser.add_argument(
        "--repo",
        default=".",
        help="Repository root used as AgentIndex base_dir (default: .)",
    )
    tree_parser.add_argument(
        "--path",
        default="",
        help="Optional path under the repository to render",
    )
    tree_parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Optional maximum tree depth",
    )

    args = parser.parse_args()
    if args.command == "tree":
        output = AgentIndex(args.db, base_dir=args.repo).tree(
            path=args.path,
            depth=args.depth,
        )
        sys.stdout.write(output)
        if output and not output.endswith("\n"):
            sys.stdout.write("\n")


if __name__ == "__main__":
    main()
