"""Markdown file walker — extracts heading structure."""

from __future__ import annotations

from pathlib import Path

from simple_agent.index.tree import TreeNode


def _walk_markdown_file(file_path: Path) -> TreeNode:
    """Walk a Markdown file and return a TreeNode with heading children.

    TODO: implement Markdown heading parsing."""
    return TreeNode(name=file_path.name, is_dir=False,
                    metadata={"abs_path": str(file_path)})
