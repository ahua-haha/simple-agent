"""TreeNode and file walker dispatcher."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any


class TreeNode:
    """A node in the in-memory tree for rendering, with name, comment, and children."""

    def __init__(self, name: str, is_dir: bool = False, comment: str = "",
                 children: list[TreeNode] | None = None,
                 metadata: dict[str, Any] | None = None):
        self.name = name
        self.is_dir = is_dir
        self.comment = comment
        self.children = children or []
        self.metadata = metadata or {}


def walk_file(
    file_path: Path,
    *,
    depth: int | None = None,
    current_depth: int = 0,
    filter_fn: Callable[[Path], bool] | None = None,
) -> TreeNode | None:
    """Walk a file and return a TreeNode with symbol children.

    Routes to type-specific walkers based on file extension.
    *depth* controls symbol nesting; shared with ``walk_dir``.
    Sets ``metadata["abs_path"]``. Does NOT access the database."""

    if filter_fn is not None and filter_fn(file_path):
        return None

    suffix = file_path.suffix
    if suffix == ".py":
        from simple_agent.index.python_walker import _walk_python_file
        return _walk_python_file(file_path, depth=depth, current_depth=current_depth)
    elif suffix in (".md", ".mdx", ".markdown"):
        from simple_agent.index.markdown_walker import _walk_markdown_file
        return _walk_markdown_file(file_path)
    else:
        return _walk_generic_file(file_path)


def _walk_generic_file(file_path: Path) -> TreeNode:
    """Walk a generic file and return a TreeNode."""
    return TreeNode(name=file_path.name, is_dir=False,
                    metadata={"abs_path": str(file_path)})


def build_tree(
    base_path: str = "",
    *,
    depth: int | None = None,
    filter_fn: Callable[[Path], bool] | None = None,
) -> TreeNode | None:
    """Walk the filesystem under *base_path*, return a ``TreeNode``.

    Does NOT access the database."""
    from simple_agent.index.dir_walker import walk_dir

    full = Path(base_path).resolve() if base_path else Path(".").resolve()
    if not full.is_dir():
        return None

    return walk_dir(full, depth=depth, filter_fn=filter_fn)


def render_tree(node: TreeNode) -> str:
    """Render a TreeNode tree as an ASCII tree.

    Pure output. Uses ``node.comment`` for the trailing annotation."""
    if node is None:
        return "(empty)"

    def _render(n: TreeNode, prefix: str = "", is_last: bool = True,
                is_root: bool = True) -> str:
        output = ""

        if is_root:
            pointer = ""
            current_prefix = ""
        else:
            pointer = "└── " if is_last else "├── "
            current_prefix = prefix + pointer

        suffix = "/" if n.is_dir else ""
        comment = f"  # {n.comment}" if n.comment else ""

        if is_root:
            label = n.metadata.get("abs_path", n.name)
            output += f"{label}{suffix}{comment}\n"
        else:
            output += f"{current_prefix}{n.name}{suffix}{comment}\n"

        if n.children:
            sorted_children = sorted(
                n.children,
                key=lambda c: (0 if c.is_dir else 1, c.name.lower())
            )

            next_prefix = prefix if is_root else prefix + ("    " if is_last else "│   ")

            num_children = len(sorted_children)
            for index, child in enumerate(sorted_children):
                is_child_last = (index == num_children - 1)
                output += _render(
                    child,
                    prefix=next_prefix,
                    is_last=is_child_last,
                    is_root=False
                )

        return output

    return _render(node)
