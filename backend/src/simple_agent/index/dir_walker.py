"""Directory walker."""

from __future__ import annotations

from pathlib import Path

from simple_agent.index.models import DirectoryNode
from simple_agent.index.tree import WalkOptions, walk_file


def walk_dir(root: str | Path, options: WalkOptions) -> DirectoryNode:
    """Walk a directory node and return *root* with children populated."""
    dir_path = Path(root)
    root_node = DirectoryNode(path=str(dir_path), name=dir_path.name or str(dir_path))
    if options.should_skip(dir_path) or options.at_depth_limit():
        return root_node

    try:
        entries = sorted(dir_path.iterdir())
    except OSError:
        return root_node

    for entry in entries:
        if options.should_skip(entry):
            continue
        child_options = options.child()
        if entry.is_dir():
            root_node.children.append(walk_dir(entry, child_options))
        elif entry.is_file():
            root_node.children.append(walk_file(entry, child_options))

    return root_node
