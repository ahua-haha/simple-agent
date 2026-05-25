"""Directory walker — walks directories into TreeNodes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from simple_agent.index.tree import TreeNode, walk_file


def walk_dir(
    dir_path: Path,
    *,
    depth: int | None = None,
    current_depth: int = 0,
    filter_fn: Callable[[Path], bool] | None = None,
) -> TreeNode | None:
    """Walk a directory and return a TreeNode with its children.

    Sets ``metadata["abs_path"]``. Does NOT access the database."""

    if filter_fn is not None and filter_fn(dir_path):
        return None

    node = TreeNode(name=dir_path.name or str(dir_path), is_dir=True,
                    metadata={"abs_path": str(dir_path)})

    if depth is not None and current_depth >= depth:
        return node

    try:
        entries = sorted(dir_path.iterdir())
    except OSError:
        return node

    for entry in entries:
        if entry.is_dir():
            child = walk_dir(entry, depth=depth,
                             current_depth=current_depth + 1,
                             filter_fn=filter_fn)
            if child is not None:
                node.children.append(child)
        elif entry.is_file():
            child = walk_file(entry, depth=depth,
                              current_depth=current_depth + 1,
                              filter_fn=filter_fn)
            if child is not None:
                node.children.append(child)

    return node
