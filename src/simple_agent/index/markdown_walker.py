"""Markdown file walker — extracts heading structure."""

from __future__ import annotations

from simple_agent.index.models import FileNode
from simple_agent.index.tree import WalkOptions


def walk_markdown_file(root: FileNode, options: WalkOptions) -> FileNode:
    """Walk a Markdown file node and return *root* with heading children populated.

    TODO: implement Markdown heading parsing.
    """
    return root
