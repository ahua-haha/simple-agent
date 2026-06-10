"""Index node walker dispatcher and renderer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from simple_agent.index.models import BaseNode, DirectoryNode, FileNode


@dataclass(frozen=True)
class WalkOptions:
    depth: int | None = None
    current_depth: int = 0
    filter_fn: Callable[[Path], bool] | None = None

    def child(self) -> "WalkOptions":
        return WalkOptions(
            depth=self.depth,
            current_depth=self.current_depth + 1,
            filter_fn=self.filter_fn,
        )

    def should_skip(self, path: Path) -> bool:
        return self.filter_fn is not None and self.filter_fn(path)

    def at_depth_limit(self) -> bool:
        return self.depth is not None and self.current_depth >= self.depth


def walk_file(root: FileNode, options: WalkOptions) -> FileNode:
    """Walk a file node and return *root* with any symbol children populated."""
    file_path = Path(root.path)
    if options.should_skip(file_path):
        return root

    suffix = file_path.suffix
    if suffix == ".py":
        from simple_agent.index.python_walker import walk_python_file
        return walk_python_file(root, options)
    if suffix in (".md", ".mdx", ".markdown"):
        from simple_agent.index.markdown_walker import walk_markdown_file
        return walk_markdown_file(root, options)
    return root


def build_tree(
    root: BaseNode,
    options: WalkOptions,
) -> BaseNode:
    """Walk from *root* and return it with children populated."""
    from simple_agent.index.dir_walker import walk_dir

    root_path = Path(root.path)
    if isinstance(root, DirectoryNode) and root_path.is_dir():
        return walk_dir(root, options)
    if isinstance(root, FileNode) and root_path.is_file():
        return walk_file(root, options)
    return root


def render_tree(node: BaseNode | None, *, depth: int | None = None) -> str:
    """Render a ``BaseNode`` tree as an ASCII tree."""
    if node is None:
        return "(empty)"

    def _render(
        current: BaseNode,
        prefix: str = "",
        is_last: bool = True,
        is_root: bool = True,
        current_depth: int = 0,
    ) -> str:
        output = ""

        if is_root:
            pointer = ""
            current_prefix = ""
        else:
            pointer = "└── " if is_last else "├── "
            current_prefix = prefix + pointer

        if is_root:
            output += f"{current.format_node(label=current.path)}\n"
        else:
            output += f"{current_prefix}{current.format_node()}\n"

        if current.children and (depth is None or current_depth < depth):
            sorted_children = sorted(
                current.children,
                key=lambda child: (0 if child.is_dir else 1, child.name.lower()),
            )

            next_prefix = prefix if is_root else prefix + ("    " if is_last else "│   ")

            num_children = len(sorted_children)
            for index, child in enumerate(sorted_children):
                is_child_last = index == num_children - 1
                output += _render(
                    child,
                    prefix=next_prefix,
                    is_last=is_child_last,
                    is_root=False,
                    current_depth=current_depth + 1,
                )

        return output

    return _render(node)
