"""Markdown file walker - builds a heading-only outline tree."""

from __future__ import annotations

import re

from pathlib import Path

from simple_agent.index.models import FileNode, SymbolNode
from simple_agent.index.tree import WalkOptions


_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")
_FENCE_RE = re.compile(r"^[ \t]{0,3}(```+|~~~+)")


def walk_markdown_file(root: str | Path, options: WalkOptions) -> FileNode:
    """Walk a Markdown file node and return *root* with heading children populated."""
    file_path = Path(root)
    root_node = FileNode(path=str(file_path), name=file_path.name)
    if options.at_depth_limit():
        return root_node

    try:
        source = file_path.read_text()
    except OSError:
        return root_node

    root_node.children = _walk_outline(source.splitlines(), options.child())
    return root_node


def _walk_outline(lines: list[str], options: WalkOptions) -> list[SymbolNode]:
    result: list[SymbolNode] = []
    heading_stack: list[tuple[int, SymbolNode, WalkOptions]] = []
    in_fence = False

    for line_number, line in enumerate(lines, start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue

        if in_fence:
            continue

        match = _HEADING_RE.match(line)
        if match is None:
            continue

        level = len(match.group(1))
        title = re.sub(r"[ \t]+#+[ \t]*$", "", match.group(2)).strip()
        if not title:
            continue

        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()

        symbol = _convert_node(title, line_number)
        if heading_stack:
            _parent_level, parent, parent_options = heading_stack[-1]
            if not parent_options.at_depth_limit():
                parent.children.append(symbol)
            heading_stack.append((level, symbol, parent_options.child()))
        else:
            if not options.at_depth_limit():
                result.append(symbol)
            heading_stack.append((level, symbol, options.child()))

    return result


def _convert_node(title: str, line_number: int) -> SymbolNode:
    label = " ".join(title.split()).replace("/", "-")
    if len(label) > 80:
        label = label[:77].rstrip() + "..."
    if not label:
        label = "heading"

    return SymbolNode(
        path=label,
        name=label,
        description="",
        line_start=line_number,
        line_end=line_number,
        symbol_type="heading",
    )
