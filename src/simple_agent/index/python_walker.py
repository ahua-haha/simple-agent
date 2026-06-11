"""Python file walker — extracts symbols using tree-sitter."""

from __future__ import annotations

from pathlib import Path

from simple_agent.index.models import FileNode, SymbolNode
from simple_agent.index.tree import WalkOptions
from typing import Any


try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser

    _PY_LANG = Language(tspython.language())
    _PY_PARSER = Parser(_PY_LANG)
except ImportError:
    _PY_LANG = None
    _PY_PARSER = None


def walk_python_file(root: str | Path, options: WalkOptions) -> FileNode:
    """Walk a Python file node and return *root* with symbol children populated."""
    file_path = Path(root)
    root_node = FileNode(path=str(file_path), name=file_path.name)
    if options.at_depth_limit() or _PY_PARSER is None:
        return root_node

    try:
        source = file_path.read_bytes()
    except OSError:
        return root_node

    tree = _PY_PARSER.parse(source)
    root_node.children = _walk_module(tree.root_node, root_node.path, options.depth)
    return root_node


def _walk_module(root: Any, file_path: str, depth: int | None) -> list[SymbolNode]:
    result: list[SymbolNode] = []

    for child in root.children:
        symbol = None

        if child.type == "class_definition":
            symbol = _walk_class_definition(child, file_path, depth, 0)

        elif child.type == "function_definition":
            symbol = _walk_function_definition(child, file_path)

        elif child.type == "decorated_definition":
            for inner in child.children:
                if inner.type == "class_definition":
                    symbol = _walk_class_definition(inner, file_path, depth, 0)
                elif inner.type == "function_definition":
                    symbol = _walk_function_definition(inner, file_path)

        if symbol is not None:
            result.append(symbol)

    return result


def _walk_class_definition(
    root: Any,
    file_path: str,
    depth: int | None,
    sym_depth: int,
) -> SymbolNode:
    cls_name = root.child_by_field_name("name")
    name = cls_name.text.decode() if cls_name else "class"
    symbol = SymbolNode(
        path=f"{file_path}:{name}",
        name=name,
        description="",
        symbol_type="class",
        line_start=_line_start(root),
        line_end=_line_end(root),
    )

    if depth is not None and sym_depth >= depth:
        return symbol

    body = root.child_by_field_name("body")
    if body is not None:
        symbol.children = _walk_block(body, symbol.path, depth, sym_depth + 1)

    return symbol


def _walk_function_definition(root: Any, parent_path: str) -> SymbolNode:
    func_name = root.child_by_field_name("name")
    name = (func_name.text.decode() + "()") if func_name else "func()"
    return SymbolNode(
        path=f"{parent_path}:{name}",
        name=name,
        description="",
        symbol_type="function",
        line_start=_line_start(root),
        line_end=_line_end(root),
    )


def _walk_block(
    root: Any,
    parent_path: str,
    depth: int | None,
    sym_depth: int,
) -> list[SymbolNode]:
    result: list[SymbolNode] = []

    for child in root.children:
        symbol = None

        if child.type == "function_definition":
            func_name = child.child_by_field_name("name")
            name = (func_name.text.decode() + "()") if func_name else "func()"
            symbol = SymbolNode(
                path=f"{parent_path}:{name}",
                name=name,
                description="",
                symbol_type="function",
                line_start=_line_start(child),
                line_end=_line_end(child),
            )

            if depth is None or sym_depth < depth:
                nested_body = child.child_by_field_name("body")
                if nested_body is not None:
                    symbol.children = _walk_block(nested_body, symbol.path, depth, sym_depth + 1)

        elif child.type == "class_definition":
            symbol = _walk_class_definition(child, parent_path, depth, sym_depth)

        if symbol is not None:
            result.append(symbol)

    return result


def _line_start(node: Any) -> int:
    return node.start_point[0] + 1


def _line_end(node: Any) -> int:
    return node.end_point[0] + 1
