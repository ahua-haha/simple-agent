"""Directory walker."""

from __future__ import annotations

from pathlib import Path

from simple_agent.index.models import DirectoryNode, FileNode
from simple_agent.index.tree import WalkOptions, walk_file


def walk_dir(root: DirectoryNode, options: WalkOptions) -> DirectoryNode:
    """Walk a directory node and return *root* with children populated."""
    dir_path = Path(root.path)
    if options.should_skip(dir_path) or options.at_depth_limit():
        return root

    try:
        entries = sorted(dir_path.iterdir())
    except OSError:
        return root

    for entry in entries:
        child_options = options.child()
        if entry.is_dir():
            child = DirectoryNode(path=str(entry))
            root.children.append(walk_dir(child, child_options))
        elif entry.is_file():
            child = FileNode(path=str(entry))
            root.children.append(walk_file(child, child_options))

    return root
